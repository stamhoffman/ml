import unittest
from cat import Cat

class TestCat(unittest.TestCase):
    def test_eat_returns_Yummy_for_fish_after_creation_with_Tom(self):
        cat = Cat('Tom')
        self.assertEqual('Yummy!', cat.eat('fish'))

    def test_eat_returns_Ugh_for_tomato_after_creation_with_Tom(self):
        cat = Cat('Tom')
        self.assertEqual('Ugh!', cat.eat('tomato'))

if __name__ == '__main__':
    unittest.main()
