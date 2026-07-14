import os
import struct
import math
import random
from cipher import PresentCipher  # Using your PRESENT cipher implementation

# Configuration
BLOCK_SIZE = 64  # 64-bit blocks
KEY_SIZE = 80    # PRESENT-64/80 configuration
CTT_STATIC_KEY = "0x00000000000000000000"  # PRESENT test vector key
PDF_FILES = ['pdfs/1.pdf', 'pdfs/2.pdf', 'pdfs/3.pdf']
OUTPUT_DIR = 'datasets'

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def bytes_to_int(b):
    return int.from_bytes(b, byteorder='big')

def int_to_bytes(n):
    return n.to_bytes(8, byteorder='big')

def generate_ctt_incremental():
    """Generate Dataset1: Incremental CTT"""
    initial = os.urandom(8)
    fixed_part = initial[:5]
    start_value = bytes_to_int(initial[5:])
    cipher = PresentCipher(CTT_STATIC_KEY, rounds=31)
    
    with open(f'{OUTPUT_DIR}/dataset1.csv', 'w') as f:
        f.write("plaintext,ciphertext\n")
        for i in range(2**15):
            var_part = (start_value + i) & 0xFFFFFF
            plaintext_bytes = fixed_part + struct.pack('>I', var_part)[1:]
            plaintext_int = bytes_to_int(plaintext_bytes)
            ciphertext = cipher.encrypt(plaintext_bytes)
            ciphertext_int = bytes_to_int(ciphertext)
            f.write(f"{plaintext_int:016x},{ciphertext_int:016x}\n")

def generate_ctt_decremental():
    """Generate Datasets 2-4: Decremental CTT"""
    for ds_num in range(2, 5):
        initial = os.urandom(8)
        fixed_part = initial[:5]
        start_value = bytes_to_int(initial[5:])
        cipher = PresentCipher(CTT_STATIC_KEY, rounds=31)
        
        with open(f'{OUTPUT_DIR}/dataset{ds_num}.csv', 'w') as f:
            f.write("plaintext,ciphertext\n")
            for i in range(2**11):
                var_part = (start_value - i) & 0xFFFFFF
                plaintext_bytes = fixed_part + struct.pack('>I', var_part)[1:]
                plaintext_int = bytes_to_int(plaintext_bytes)
                ciphertext = cipher.encrypt(plaintext_bytes)
                ciphertext_int = bytes_to_int(ciphertext)
                f.write(f"{plaintext_int:016x},{ciphertext_int:016x}\n")

def generate_nctt():
    """Generate Datasets 5-7: Non-correlated from PDFs"""
    sizes = [int(2**16.3), int(2**14.6), int(2**15.3)]
    
    for i, (pdf, size) in enumerate(zip(PDF_FILES, sizes), start=5):
        # Generate unique random key per dataset
        key = "0x" + os.urandom(10).hex()  # 80-bit key (10 bytes)
        cipher = PresentCipher(key, rounds=31)
        
        # Read and process PDF
        with open(pdf, 'rb') as f:
            data = f.read()
        
        # Pad data to multiple of 8 bytes
        if len(data) % 8 != 0:
            data += b'\x00' * (8 - len(data) % 8)
        
        # Process chunks
        seen = set()
        count = 0
        with open(f'{OUTPUT_DIR}/dataset{i}.csv', 'w') as out:
            out.write("plaintext,ciphertext\n")
            for j in range(0, len(data), 8):
                chunk = data[j:j+8]
                if chunk in seen:
                    continue
                seen.add(chunk)
                
                plaintext_int = bytes_to_int(chunk)
                ciphertext = cipher.encrypt(chunk)
                ciphertext_int = bytes_to_int(ciphertext)
                out.write(f"{plaintext_int:016x},{ciphertext_int:016x}\n")
                
                count += 1
                if count >= size:
                    break

def main():
    create_dir(OUTPUT_DIR)
    print("Generating dataset1 (Incremental CTT)...")
    generate_ctt_incremental()
    
    print("Generating datasets 2-4 (Decremental CTT)...")
    generate_ctt_decremental()
    
    print("Generating datasets 5-7 (Non-correlated NCTT)...")
    generate_nctt()
    
    print("All datasets generated successfully!")

if __name__ == "__main__":
    main()