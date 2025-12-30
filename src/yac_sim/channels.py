class GilbertElliottChannel:
    """
    Two-state Markov channel.
    state=0: Good; state=1: Bad
    - p_loss_good: loss prob in Good
    - p_loss_bad: loss prob in Bad
    - p_g2b: P(Good->Bad)
    - p_b2g: P(Bad->Good)
    """

    def __init__(
        self,
        rng,
        p_loss_good=0.05,
        p_loss_bad=0.5,
        p_g2b=0.02,
        p_b2g=0.1,
        init_state=0,
    ):
        self.rng = rng
        self.p_loss_good = p_loss_good
        self.p_loss_bad = p_loss_bad
        self.p_g2b = p_g2b
        self.p_b2g = p_b2g
        self.state = init_state

    def step(self):
        if self.state == 0:
            if self.rng.random() < self.p_g2b:
                self.state = 1
        else:
            if self.rng.random() < self.p_b2g:
                self.state = 0

    def transmit(self):
        """Return received(bool), loss_prob(float), state(int)."""
        p = self.p_loss_good if self.state == 0 else self.p_loss_bad
        received = self.rng.random() > p
        return received, p, self.state
