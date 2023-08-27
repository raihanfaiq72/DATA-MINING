# Jalankan instalasi library berikut jika belum terinstall
# pip install google-api-python-client

import csv
from googleapiclient.discovery import build

# Set up the API key and YouTube Data API service
## Masukkan API key yang sudah dibuat di Google Cloud Platform
## Contoh : AIzaSyCFjru8dOZbGtZUi_AQu1Cz1MLoANaY22k
API_KEY = "AIzaSyCFjru8dOZbGtZUi_AQu1Cz1MLoANaY22k"
youtube = build("youtube", "v3", developerKey=API_KEY)

def scrape_comments(video_id):
    # Get the video details
    video_response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    video_title = video_response['items'][0]['snippet']['title']
    print("Scraping comments for video:", video_title)

    # Get the video comments
    comments = []
    next_page_token = None

    while True:
        comment_response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        ).execute()

        for item in comment_response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = comment_response.get("nextPageToken")

        if not next_page_token:
            break

    return comments

# Test the function
## Masukkan ID video youtube yang ingin diambil komentarnya
## Contoh : 5kAF9QV5nYQ
video_id = "5kAF9QV5nYQ"
comments = scrape_comments(video_id)

# Save comments to CSV file
filename = "comments.csv"
with open(filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Comment"])
    writer.writerows(zip(comments))

print("Comments saved to", filename)