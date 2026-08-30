# test_contract.py -- quick sanity check, not a real test suite yet
from contract import Stream

class DummyStream(Stream):
    def score(self, window):
        self.last_quality = 1.0
        return 0.5

s = DummyStream()
print(s.score({}))          # -> 0.5
print(s.last_quality)       # -> 1.0

# Confirm the ABC actually blocks incomplete implementations:
try:
    Stream()
    print("FAIL: Stream should not be instantiable directly")
except TypeError:
    print("PASS: Stream correctly abstract")