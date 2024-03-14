import random
import string

def generate_otp(length=6):
    # Generate a random string of digits
    digits = string.digits
    otp = ''.join(random.choice(digits) for _ in range(length))
    return otp

# Example usage:
otp = generate_otp()
print("Generated OTP:", otp)
