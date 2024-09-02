import random 

from pedalboard import (
    Pedalboard,
    Distortion,
    Compressor,
    Delay,
    Chorus, 
    Reverb,
    Gain
)

class GuitarEffects:
    
    def __init__(self, 
                 n_random_effects,
                 gain_prob=0.5,
                 distortion_prob=0.5,
                 compressor_prob=0.5,
                 delay_prob=0.5,
                 chorus_prob=0.5,
                 reverb_prob=0.5):
        self.effects = []
        for i in range(n_random_effects):
            pedals = []
            if random.random() < gain_prob:
                pedals.append(Gain(
                    gain_db=random.uniform(0, 50)
                ))
            if random.random() < distortion_prob:
                pedals.append(Distortion(
                    drive_db=random.uniform(0, 50)
                ))
            if random.random() < compressor_prob:
                pedals.append(Compressor(
                    threshold_db=random.uniform(0, 1),
                    ratio=random.uniform(1, 10),
                    attack_ms=random.uniform(0, 1),
                    release_ms=random.uniform(0, 1000)
                ))
            if random.random() < delay_prob:
                pedals.append(Delay(
                    delay_seconds=random.uniform(0, 1),
                    feedback=random.uniform(0, 1), 
                    mix=random.uniform(0, 1)
                ))
            if random.random() < chorus_prob:
                pedals.append(Chorus(
                    rate_hz=random.uniform(0, 1),
                    depth=random.uniform(0, 1),
                    centre_delay_ms=random.uniform(0, 1),
                    feedback=random.uniform(0, 1), 
                    mix=random.uniform(0, 1)
                ))
            if random.random() < reverb_prob:
                pedals.append(Reverb(
                    room_size=random.uniform(0, 1),
                    damping=random.uniform(0, 1),
                    wet_level=random.uniform(0, 1),
                    width=random.uniform(0, 1),
                    dry_level=random.uniform(0, 1)
                ))
            board = Pedalboard(pedals)
            self.effects.append(board)
        

    def __call__(self, audio, sample_rate):
        return self.effects[random.randint(0, len(self.effects)-1)](audio, sample_rate)