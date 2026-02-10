"""Run ONLY the backward compat test to check if it's a pre-existing failure."""
import sys, os
os.chdir(r"S:\AI\work\VRAXION_DEV\Golden Draft")
sys.path.insert(0, "tests")
import unittest
loader = unittest.TestLoader()
suite = unittest.TestSuite()
suite.addTest(loader.loadTestsFromName(
    "test_fibonacci_swarm.FibonacciSwarmTests.test_fibonacci_disabled_backward_compat"
))
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
