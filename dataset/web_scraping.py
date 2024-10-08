import requests
from bs4 import BeautifulSoup
import csv

# URL to the main page listing all countries
url = 'https://pasparetbloggen.blogspot.com/p/blog-page.html'

# Fetch the content from the main page
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

with open('country_points.csv', mode='w', newline='', encoding='utf-8') as csv_file:
    # Define columns with "points" instead of "poäng"
    fieldnames = ['Country', '10 points', '8 points', '6 points', '4 points', '2 points']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Find the "Världen" section
    varlden_section = soup.find('b', string='Världen')

    # Process the countries listed under "Världen"
    sibling = varlden_section.find_next_sibling()
    while sibling:
        # If 'div', find all links; otherwise, treat it as a direct link
        country_links = sibling.find_all('a') if sibling.name == 'div' else [sibling]

        for country_link in country_links:
            country_name = country_link.get_text(strip=True)
            country_url = country_link.get('href')

            # Check if the link is valid
            if not country_url:
                continue

            # Ensure the URL is absolute
            if not country_url.startswith('http'):
                country_url = 'https://pasparetbloggen.blogspot.com' + country_url

            # Initialize an empty entry for each country with blank clues
            country_data = {
                'Country': country_name,
                '10 points': '',
                '8 points': '',
                '6 points': '',
                '4 points': '',
                '2 points': ''
            }

            # Write the country name with empty clue columns to the CSV
            csv_writer.writerow(country_data)

        # Move to the next sibling element
        sibling = sibling.find_next_sibling()

print("success")