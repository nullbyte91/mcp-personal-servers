
import unittest
from sample_slow_code import find_duplicates

class TestFindDuplicates(unittest.TestCase):
    def test_find_duplicates(self):
        # Test with an empty list
        self.assertEqual(find_duplicates([]), [])
        
        # Test with no duplicates
        self.assertEqual(sorted(find_duplicates([1, 2, 3, 4])), [])
        
        # Test with duplicates
        self.assertEqual(sorted(find_duplicates([1, 2, 2, 3, 4, 4])), [2, 4])
        
        # Test with all duplicates
        self.assertEqual(sorted(find_duplicates([1, 1, 1])), [1])

if __name__ == "__main__":
    unittest.main()
