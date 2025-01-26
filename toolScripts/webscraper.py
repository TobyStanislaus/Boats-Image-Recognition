import requests
from bs4 import BeautifulSoup
import csv
import os
import re

# Define the range of years and base URL
base_url = 'https://www.whitefriarssc.org/racing/results-'
years = range(2019, 2026)
database_path = 'database'

# Create the database directory if it doesn't exist
os.makedirs(database_path, exist_ok=True)

# Initialize a set to store unique results
unique_results = set()

# Function to clean text by removing brackets and specific patterns
def clean_text(text):
    # Remove anything in brackets
    text = re.sub(r'\(.*?\)', '', text)
    # Remove occurrences of "R" followed by numbers
    text = re.sub(r'R\d+', '', text)
    return text.strip()

# Loop through each year
for year in years:
    # Construct the URL for the current year
    url = f"{base_url}{year}"

    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the tables on the page
    tables = soup.find_all('table')

    # Loop through each table
    for table in tables:
        # Extract the headers to map the column indices
        headers = [header.get_text(strip=True).lower() for header in table.find_all('th')]

        # Dynamically determine column indices based on headers
        try:
            class_idx = headers.index('class')
            boat_number_idx = headers.index('sailno')
            helm_name_idx = headers.index('helmname')
            crew_name_idx = headers.index('crewname')
        except ValueError:
            # Skip this table if it doesn't have the required headers
            continue

        # Extract rows
        for row in table.find_all('tr')[1:]:  # Skip the header row
            cells = row.find_all('td')
            if len(cells) >= len(headers):  # Ensure row has enough columns
                # Clean and process each cell's text
                boat_class = clean_text(cells[class_idx].get_text(strip=True))
                boat_number = clean_text(cells[boat_number_idx].get_text(strip=True))
                helm_name = clean_text(cells[helm_name_idx].get_text(strip=True))
                crew_name = clean_text(cells[crew_name_idx].get_text(strip=True) if crew_name_idx < len(cells) else '')

                # Add the result as a tuple to the set to avoid duplicates
                unique_results.add((boat_class, boat_number, helm_name, crew_name))

# Define the output CSV file path
output_file = os.path.join(database_path, 'sailing_results.csv')

# Write the results to a CSV file
with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Class', 'Boat Number', 'Helm Name', 'Crew Name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Write each unique result
    for result in unique_results:
        writer.writerow({
            'Class': result[0],
            'Boat Number': result[1],
            'Helm Name': result[2],
            'Crew Name': result[3]
        })

print(f"Scraping complete. Results saved to {output_file}")
