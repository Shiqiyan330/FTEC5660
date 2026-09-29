import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import hw1


DIRECT = {"amount_paid": "102.30", "subtotal": "102.31"}
ITEMS = {
    "item_amounts": ["10.00", "36.90", "60.80"],
    "discounts": ["5.39"],
    "rounding": "-0.01",
}


class FakeChain:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


class ReceiptValidationTests(unittest.TestCase):
    def setUp(self):
        self.image = Path(__file__).resolve().parent / "receipt.png"
        image_encoder = patch.object(hw1, "image_data_url", return_value="data:image/png;base64,dGVzdA==")
        image_encoder.start()
        self.addCleanup(image_encoder.stop)

    def test_rounding_example_and_separate_calls(self):
        chain = FakeChain([DIRECT, ITEMS])
        self.assertEqual(hw1.answer_queries(chain, [self.image]), {
            hw1.QUERY_1: "HK$102.30", hw1.QUERY_2: "HK$107.70",
        })
        self.assertEqual(len(chain.calls), 2)
        messages = [call["receipt_messages"][0].content for call in chain.calls]
        self.assertIn("TOTAL", messages[0][0]["text"])
        self.assertIn("ITEMS", messages[1][0]["text"])
        self.assertEqual(messages[0][1], messages[1][1])

    def test_decimal_precision_no_discount(self):
        chain = FakeChain([
            {"amount_paid": "0.30", "subtotal": "0.30"},
            {"item_amounts": ["0.10", "0.20"], "discounts": [], "rounding": "0"},
        ])
        self.assertEqual(hw1.answer_queries(chain, [self.image])[hw1.QUERY_1], "HK$0.30")

    def test_multiple_receipts_and_positive_rounding(self):
        chain = FakeChain([
            DIRECT, ITEMS,
            {"amount_paid": "7.00", "subtotal": "6.99"},
            {"item_amounts": ["10"], "discounts": ["2", "1.01"], "rounding": "0.01"},
        ])
        result = hw1.answer_queries(chain, [self.image, self.image])
        self.assertEqual(result, {hw1.QUERY_1: "HK$109.30", hw1.QUERY_2: "HK$117.70"})

    def test_mismatch_retries_without_double_counting(self):
        bad = {**ITEMS, "discounts": []}
        chain = FakeChain([DIRECT, bad, DIRECT, ITEMS])
        self.assertEqual(hw1.answer_queries(chain, [self.image])[hw1.QUERY_1], "HK$102.30")
        self.assertEqual(len(chain.calls), 4)
        self.assertIn("previous attempt", chain.calls[2]["receipt_messages"][0].content[0]["text"])

    def test_third_retry_can_succeed(self):
        bad = {**ITEMS, "discounts": []}
        chain = FakeChain([DIRECT, bad] * 3 + [DIRECT, ITEMS])
        self.assertEqual(hw1.answer_queries(chain, [self.image])[hw1.QUERY_2], "HK$107.70")
        self.assertEqual(len(chain.calls), 8)

    def test_exhausted_retries_raise(self):
        chain = FakeChain([DIRECT, {**ITEMS, "discounts": []}] * 4)
        with self.assertRaisesRegex(ValueError, "receipt.png.*after 4 attempts"):
            hw1.answer_queries(chain, [self.image])
        self.assertEqual(len(chain.calls), 8)

    def test_matching_paid_but_wrong_subtotal_is_rejected(self):
        chain = FakeChain([{**DIRECT, "subtotal": "102.30"}, ITEMS] * 4)
        with self.assertRaisesRegex(ValueError, "amounts do not match"):
            hw1.answer_queries(chain, [self.image])

    def test_invalid_values_retry_and_recover(self):
        for value in [None, True, "NaN", "Infinity", "-1.00", "1.001", "unreadable"]:
            with self.subTest(value=value):
                chain = FakeChain([{**DIRECT, "amount_paid": value}, ITEMS, DIRECT, ITEMS])
                self.assertEqual(hw1.answer_queries(chain, [self.image])[hw1.QUERY_1], "HK$102.30")
                self.assertEqual(len(chain.calls), 4)

    def test_missing_fields_and_bad_lists_are_rejected(self):
        for direct, items in [
            ([], ITEMS), ({}, ITEMS), (DIRECT, None),
            (DIRECT, {**ITEMS, "discounts": "5.39"}),
            (DIRECT, {**ITEMS, "item_amounts": []}),
            (DIRECT, {**ITEMS, "rounding": None}),
        ]:
            with self.subTest(direct=direct, items=items):
                chain = FakeChain([direct, items] * 4)
                with self.assertRaises(ValueError):
                    hw1.answer_queries(chain, [self.image])
                self.assertEqual(len(chain.calls), 8)

    def test_invalid_json_retries(self):
        from langchain_core.exceptions import OutputParserException
        chain = FakeChain([OutputParserException("invalid JSON"), DIRECT, ITEMS])
        self.assertEqual(hw1.answer_queries(chain, [self.image])[hw1.QUERY_2], "HK$107.70")
        self.assertEqual(len(chain.calls), 3)

    def test_failed_receipt_prevents_csv_output(self):
        import argparse
        chain = FakeChain([DIRECT, ITEMS] + [DIRECT, {**ITEMS, "discounts": []}] * 4)
        with (
            patch.object(hw1, "parse_args", return_value=argparse.Namespace(image_folder=self.image.parent)),
            patch.object(hw1, "image_files", return_value=[self.image, self.image]),
            patch.object(hw1, "load_env_file"),
            patch.object(hw1, "build_chain", return_value=chain),
            patch.object(hw1, "write_results") as write,
        ):
            with self.assertRaises(ValueError):
                hw1.main()
            write.assert_not_called()

    def test_transport_error_is_not_misreported_as_amount_mismatch(self):
        chain = FakeChain([ConnectionError("offline")])
        with self.assertRaises(ConnectionError):
            hw1.answer_queries(chain, [self.image])
        self.assertEqual(len(chain.calls), 1)

    def test_real_prompt_and_parser_pipeline_writes_csv(self):
        import argparse
        import csv
        import io
        import json
        from contextlib import contextmanager
        from decimal import Decimal
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        received = []

        def model_response(prompt):
            messages = prompt.to_messages()
            received.append(messages)
            method = messages[-1].content[0]["text"]
            result = DIRECT if "TOTAL" in method else ITEMS
            return AIMessage(content=json.dumps(result))

        csv_buffer = io.StringIO()

        @contextmanager
        def csv_output(*args, **kwargs):
            yield csv_buffer

        with patch("langchain_deepseek.ChatDeepSeek", return_value=RunnableLambda(model_response)) as model:
            chain = hw1.build_chain()
            model.assert_called_once_with(
                model="deepseek-v4-flash-vision-exp",
                temperature=0, timeout=60, max_retries=0,
            )
        with (
            patch.object(hw1, "parse_args", return_value=argparse.Namespace(image_folder=self.image.parent)),
            patch.object(hw1, "image_files", return_value=[self.image]),
            patch.object(hw1, "load_env_file"),
            patch.object(hw1, "build_chain", return_value=chain),
            patch.object(hw1, "read_ground_truth", return_value={
                hw1.QUERY_1: Decimal("102.30"), hw1.QUERY_2: Decimal("107.70"),
            }),
            patch.object(Path, "open", side_effect=csv_output),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            self.assertEqual(hw1.main(), 0)
        rows = list(csv.DictReader(io.StringIO(csv_buffer.getvalue())))
        self.assertEqual([r["query"] for r in rows], list(hw1.QUERIES))
        self.assertEqual([r["model_response"] for r in rows], ["HK$102.30", "HK$107.70"])
        self.assertEqual([r["correctness"] for r in rows], ["correct", "correct"])
        self.assertEqual(len(received), 2)
        self.assertTrue(all(len(messages) == 2 for messages in received))
        self.assertTrue(all(messages[0].type == "system" for messages in received))


if __name__ == "__main__":
    unittest.main()
