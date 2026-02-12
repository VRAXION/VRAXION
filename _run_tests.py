import sys, os
os.chdir(r"S:\AI\work\VRAXION_DEV\Golden Draft")
sys.path.insert(0, "tests")
import unittest
loader = unittest.TestLoader()
suite = unittest.TestSuite()
# All Fibonacci Swarm tests (original + byte waveform).
suite.addTests(loader.loadTestsFromName("test_fibonacci_swarm"))
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
