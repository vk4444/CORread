import pytest
import torch

from corread.models import (
    DENSE_UNITS,
    WORD_UNITS,
    CORnetS,
    CORnetZ,
    CORnetZBiasedWords,
    CORnetZNonbiasedWords,
    build_network,
    network_class,
)

IMAGES = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize('choice', ['z', 's'])
@pytest.mark.parametrize(('mode', 'outputs'), [('pre', 1000), ('lit_bias', 2000), ('lit_no_bias', 2000)])
def test_every_network_returns_all_activations(choice, mode, outputs):
    net = build_network(choice, mode).eval()
    with torch.no_grad():
        v1, v2, v4, it, h, out = net(IMAGES[:1])
    assert h.shape == (1, DENSE_UNITS)
    assert out.shape == (1, outputs)
    assert all(a.dim() == 2 for a in (v1, v2, v4, it))


def test_output_sizes_are_configurable():
    assert build_network('z', 'pre', img_classes=10).decoder.linear.out_features == 10
    net = build_network('z', 'lit_no_bias', img_classes=10, wrd_classes=5)
    assert net.linear.out_features == 15


def test_literate_network_continues_the_pre_literate_one():
    torch.manual_seed(0)
    pre = CORnetZ()
    for cls in (CORnetZBiasedWords, CORnetZNonbiasedWords):
        net = cls(pre)
        assert net.V1 is pre.V1 and net.IT is pre.IT
        readout = net.bilinear.lin_img if cls is CORnetZBiasedWords else net.linear
        assert torch.equal(readout.weight[:1000], pre.decoder.linear.weight)
        assert torch.equal(readout.bias[:1000], pre.decoder.linear.bias)
        # so the image outputs are those of the pre-literate network
        pre.eval()
        net.eval()
        with torch.no_grad():
            assert torch.allclose(net(IMAGES)[-1][:, :1000], pre(IMAGES)[-1])


def test_biased_word_outputs_read_only_the_word_channel():
    net = CORnetZBiasedWords().eval()
    h = torch.randn(3, DENSE_UNITS)
    changed = h.clone()
    changed[:, : DENSE_UNITS - WORD_UNITS] += 1
    with torch.no_grad():
        assert torch.equal(net.readout(h)[:, 1000:], net.readout(changed)[:, 1000:])
        assert not torch.equal(net.readout(h)[:, :1000], net.readout(changed)[:, :1000])


def test_state_dict_names():
    # the names checkpoints are saved under; changing them would break every saved model
    keys = set(CORnetZBiasedWords().state_dict())
    assert {'V1.conv.weight', 'IT.conv.bias', 'bilinear.lin_img.weight', 'bilinear.lin_wrd.weight'} <= keys
    assert 'decoder.linear.weight' in CORnetS().state_dict()
    assert 'linear.weight' in CORnetZNonbiasedWords().state_dict()


def test_network_class_rejects_unknown():
    assert network_class('s', 'illit') is CORnetS
    with pytest.raises(ValueError):
        network_class('x', 'pre')
