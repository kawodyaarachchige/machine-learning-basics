import random
import csv
import string

# Lists for generating realistic data
first_names = ["S", "A R", "M M", "W M", "E A", "K P", "R T", "N S", "J M", "P L", "T R", "H K", "L M", "S T", "D P", "C S", "A T"]
last_names = ["Jayarathne", "Pradeep", "Prasanna", "Krishantha", "Hemantha", "Wijesinghe", "Fernando", "Perera", "Silva", "Gunawardena", "Bandara", "Rajapaksa", "Dissanayake", "Mendis", "Karunaratne", "Weerasinghe", "Jayawardena"]
streets = ["Ulhitiya", "Samagimawatha", "Napatavela", "Mahawawa", "Madakadura", "Kandy Road", "Hill Street", "Main Street", "Lake View", "Park Road", "Temple Road", "Church Lane", "River Road", "School Lane", "Station Road", "Garden Lane", "Central Road"]
cities = ["Girandurukotte", "Kudagama", "Padiyapalalla", "Mathurata", "Munwatta", "Hatton", "Balangoda", "Gampola", "Nuwara Eliya", "Pelmadulla", "Boralanda", "Opanayaka", "Ruvaneliya"]
block_prefixes = ["No ", "D ", "L", "A/", "B/", "C/"]

# Function to generate unique National ID (9 or 12 digits)
def generate_national_id(used_ids):
    while True:
        if random.choice([True, False]):
            # 9-digit format: e.g., 791864690V
            id_num = f"{random.randint(100000000, 999999999)}V"
        else:
            # 12-digit format: e.g., 198728103039
            id_num = f"{random.randint(100000000000, 999999999999)}"
        if id_num not in used_ids:
            used_ids.add(id_num)
            return id_num

# Function to generate unique phone number
def generate_phone_number(used_numbers):
    while True:
        phone = f"7{random.randint(10000000, 99999999)}"
        if phone not in used_numbers:
            used_numbers.add(phone)
            return phone

# Function to generate account number (7 to 15 digits)
def generate_account_number(used_accounts):
    while True:
        length = random.randint(7, 15)
        account = "".join(random.choices(string.digits, k=length))
        if account not in used_accounts:
            used_accounts.add(account)
            return account

# Generate 200 rows of data
data = []
used_national_ids = set()
used_sms_numbers = set()
used_account_numbers = set()

for _ in range(500):
    national_id = generate_national_id(used_national_ids)
    sms_no = generate_phone_number(used_sms_numbers)
    # WhatsApp number is either the same as SMS or different
    whatsapp_no = sms_no if random.choice([True, False]) else generate_phone_number(used_sms_numbers)
    account_number = generate_account_number(used_account_numbers)
    
    row = {
        "National ID": national_id,
        "Mode": "physical",
        "Joined Date": "23/05/2025",
        "First Name": random.choice(first_names),
        "Last Name": random.choice(last_names),
        "Block No": f"{random.choice(block_prefixes)}{random.randint(1, 500)}{random.choice(['', '/A', '/B', '/C'])}",
        "Street": random.choice(streets),
        "City": random.choice(cities),
        "District": "Nuwara Eliya",
        "SMS No": 728893383,
        "WhatsApp No": whatsapp_no,
        "Territory Code": "TR029",
        "Area Code": "AR04",
        "Bank Code": "7010",
        "Branch Code": "BOC-515",
        "Language": "ENGLISH",
        "Category": "LL",
        "Account Number": account_number,
        "Electrician Grade": "D"
    }
    data.append(row)

# Write to CSV
with open("electrician_data.csv", "w", newline="") as csvfile:
    fieldnames = ["National ID", "Mode", "Joined Date", "First Name", "Last Name", "Block No", "Street", "City", "District", 
                  "SMS No", "WhatsApp No", "Territory Code", "Area Code", "Bank Code", "Branch Code", "Language", "Category", 
                  "Account Number", "Electrician Grade"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print("Generated 500 rows and saved to electrician_data.csv")