import torch.nn as nn

class SDFModel(nn.Module):
    ''' SDFModel model class.

    Args:
        device (device): torch device
        discriminator (nn.Module): discriminator network
        generator (nn.Module): generator network
        generator_test (nn.Module): generator_test network
    '''

    def __init__(self, device=None, encoder=None,
                 discriminator=None, generator=None, generator_test=None,
                 **kwargs):
        super().__init__()

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        if discriminator is not None:
            self.discriminator = discriminator.to(device)
        else:
            self.discriminator = None
        if generator is not None:
            self.generator = generator.to(device)
        else:
            self.generator = None

        if generator_test is not None:
            self.generator_test = generator_test.to(device)
        else:
            self.generator_test = None

    def forward(self, batch_size, **kwargs):
        gen = self.generator_test
        if gen is None:
            gen = self.generator
        return gen(batch_size=batch_size)

    def generate_test_images(self):
        gen = self.generator_test
        if gen is None:
            gen = self.generator
        return gen()

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model