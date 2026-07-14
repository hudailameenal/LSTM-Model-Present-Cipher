import numpy as np

# PRESENT S-box (4-bit substitution)
SBOX = [
    0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
    0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
]

# PRESENT P-box (bit permutation)
def create_pbox():
    pbox = [0] * 64
    for i in range(64):
        pbox[i] = (16 * i) % 63
    pbox[63] = 63  # Special case for the last position
    return pbox

PBOX = create_pbox()
INV_PBOX = [PBOX.index(i) for i in range(64)]  # Inverse P-box for decryption

class PresentCipher:
    def __init__(self, key_hex="0x00000000000000000000", rounds=31):
        # Remove '0x' prefix and pad to 20 hex digits (80 bits)
        key_str = key_hex[2:].zfill(20)
        self.key = int(key_str, 16)
        self.rounds = rounds
        self.round_keys = self._key_schedule()
    
    def _key_schedule(self):
        """Generate 32 round keys (64-bit each) for PRESENT-80"""
        round_keys = []
        key = self.key
        
        for i in range(1, 33):  # 32 rounds
            # 1. Extract round key (64 MSBs)
            round_keys.append((key >> 16) & 0xFFFFFFFFFFFFFFFF)
            
            # 2. Rotate key left by 61 bits
            key = ((key << 61) & ((1 << 80) - 1)) | (key >> 19)
            
            # 3. Apply S-box to leftmost 4 bits
            ms_nibble = (key >> 76) & 0xF
            updated_nibble = SBOX[ms_nibble]
            key = (key & ~(0xF << 76)) | (updated_nibble << 76)
            
            # 4. XOR with round counter (bits 15-19)
            key ^= i << 15
        
        return round_keys

    @staticmethod
    def _sbox_layer(state):
        """Apply S-box to all 16 nibbles"""
        result = 0
        for i in range(16):
            nibble = (state >> (i * 4)) & 0xF
            result |= SBOX[nibble] << (i * 4)
        return result

    @staticmethod
    def _pbox_layer(state):
        """Apply bit permutation using PBOX"""
        permuted = 0
        for i in range(64):
            bit = (state >> i) & 1
            permuted |= bit << PBOX[i]
        return permuted

    def encrypt(self, plaintext, num_rounds=None):
        """Encrypt with specified number of rounds (default: self.rounds)"""
        if num_rounds is None:
            num_rounds = self.rounds
            
        # Convert to integer
        if isinstance(plaintext, bytes):
            state = int.from_bytes(plaintext, 'big')
        else:
            state = int.from_bytes(plaintext.ljust(8, b'\x00'), 'big')
        
        # Apply the specified number of rounds
        for i in range(num_rounds):
            state ^= self.round_keys[i]
            if i < num_rounds - 1:  # Don't apply S-box and P-box in the final round
                state = self._sbox_layer(state)
                state = self._pbox_layer(state)
        
        return state.to_bytes(8, 'big')

    @staticmethod
    def encrypt_multiple_rounds(plaintext, round_keys, num_rounds):
        """Encrypt with exactly num_rounds using provided round keys"""
        if isinstance(plaintext, bytes):
            state = int.from_bytes(plaintext, 'big')
        else:
            state = int.from_bytes(plaintext.ljust(8, b'\x00'), 'big')
        
        # Apply all rounds
        for i in range(num_rounds):
            state ^= round_keys[i]
            if i < num_rounds - 1:  # Don't apply S-box and P-box in the final round
                state = PresentCipher._sbox_layer(state)
                state = PresentCipher._pbox_layer(state)
        
        return state.to_bytes(8, 'big')

    def generate_dataset(self, size, dataset_type="incremental"):
        """Generate plaintexts as per paper's specifications"""
        plaintexts = []
        
        if dataset_type == "incremental":
            # Fixed 5 bytes + incremental 3 bytes
            base = np.random.bytes(5)
            for i in range(size):
                variable = i.to_bytes(3, 'big')
                plaintexts.append(base + variable)
                
        elif dataset_type == "decremental":
            # Fixed 5 bytes + decremental 3 bytes
            base = np.random.bytes(5)
            for i in range(size):
                variable = (size - i).to_bytes(3, 'big')
                plaintexts.append(base + variable)
                
        return plaintexts

    @staticmethod
    def test():
        """Test with known PRESENT test vector"""
        cipher = PresentCipher(key_hex="0x00000000000000000000")
        plaintext = bytes.fromhex("0000000000000000")
        expected = bytes.fromhex("5579c1387b228445")
        result = cipher.encrypt(plaintext)
        
        if result == expected:
            print("PRESENT Implementation CORRECT! Test vector matches.")
        else:
            print(f"ERROR! Expected {expected.hex()}, got {result.hex()}")
        
        # Test multi-round encryption
        result1 = cipher.encrypt(plaintext, 1)
        result2 = cipher.encrypt(plaintext, 2)
        if result1 != result2:
            print("Multi-round encryption is working correctly.")
        else:
            print("ERROR: Multi-round encryption is not working!")
            
        return result == expected