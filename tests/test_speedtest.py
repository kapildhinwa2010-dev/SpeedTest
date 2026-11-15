import pytest
import torch
from speedtest import SpeedTest, create_speedtest

class TestSpeedTest:
    def test_init(self):
        m = SpeedTest(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = SpeedTest(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_speedtest()
        assert isinstance(m, SpeedTest)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
