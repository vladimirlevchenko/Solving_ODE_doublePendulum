from exp_decay import ExponentialDecay
import nose.tools as nt

def test_call():
    u = ExponentialDecay(0.4)
    nt.assert_equal(round(u(0, 3.2), 3), -1.28)

if __name__ == '__main__':
    import nose
    nose.run()
#finished comment
