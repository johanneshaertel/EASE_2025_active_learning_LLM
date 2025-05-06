import numpy as np
import requests
import json
import time
import config
import os
from datetime import datetime
import boto3
import botocore
from multiprocessing import Pool

# ----------------------------------------------------
# Decription
# ----------------------------------------------------
# New version a crawler to run on AWS and randomly selects repositories storing pulls on s3.
# ----------------------------------------------------
# Notes.
# ----------------------------------------------------
# - Currenlty other metadata is not persisted.
# - Parts may be made incremental like by using "lastEditedAt" field to see when stuff needs and update.

# ----------------------------------------------------
# Helpers.
# ----------------------------------------------------

rotation = 0

def access(query):
    
    global rotation

    unblock = 0

    while True:
        unblock = unblock + 1
        rotation = rotation + 1
        headers = {"Authorization": "token %s" %
                   config.tokens[rotation % len(config.tokens)]}
        try:
            response = requests.post(
                'https://api.github.com/graphql', json={'query': query}, headers=headers, timeout=10)

            if response.status_code == 401:
                print("Token expired " + str(config.tokens[rotation % len(config.tokens)]))

            if response.status_code == 200:
                result = response.json()

                # Check if data field
                if "data" in result:
                    return result
            
        except requests.exceptions.Timeout:
            print("Timed out")
        
        # Only sleep after a full rotation.
        if unblock % len(config.tokens) == 0:
            print("sleeping")
            time.sleep(300)  # Sleep if failed.

            if unblock > 120:
                # After two hours, unblock.
                print("unblock:-(")
                return None

def access_repository(low, high, limit = 0):
    return access('''
        query {
            search(query: "is:public language:java stars:>100 created:$low..$high", type: REPOSITORY, first: $limit) {
                repositoryCount
                edges {
                        node {
                            ... on Repository {
                                nameWithOwner
                        }
                    }
                }
            }
        }
        '''.replace("$limit", str(limit))
        .replace("$low", datetime.fromtimestamp(low).strftime('%Y-%m-%d'))
        .replace("$high", datetime.fromtimestamp(high).strftime('%Y-%m-%d')))

def access_random_repository():
    low = 1100000000
    high = datetime.timestamp(datetime.now())

    while True:
         # split
        mid = (low + high) / 2

        left = access_repository(low, mid)
        right = access_repository(mid + 24*60*60, high)

        leftcount = left["data"]["search"]["repositoryCount"]
        rightcount = right["data"]["search"]["repositoryCount"]
        
        print(datetime.fromtimestamp(low).strftime('%Y-%m-%d') + " to " + datetime.fromtimestamp(high).strftime('%Y-%m-%d') + " (" + str(leftcount) + " vs " + str(rightcount) + ")")

        # descide for left or right depending on probs
        leftprob = float(leftcount) / (leftcount + rightcount)

        if np.random.rand() < leftprob:
            low = low
            high = mid
        else:
            low = mid + 24*60*60
            high = high

        if(leftcount + rightcount < 50):
            results = access_repository(low, high, 100)
            repos = [x["node"]["nameWithOwner"] for x in results["data"]["search"]["edges"]]
            repo = np.random.choice(repos)
            return repo

def access_repository_pull_numbers(repo):
    numbers = []
    cursor = "null"
    owner = repo.split("/")[0]
    name = repo.split("/")[1]
    while True:
        data = access("""
            {repository(owner:"$owner", name:"$name") {
                    pullRequests (first:100, after: $cursor){
                    nodes{
                        number
                    }
                    pageInfo {
                        endCursor
                        hasNextPage
                    }
                }
            }}
        """.replace("$owner", owner).replace("$name", name).replace("$cursor", cursor))

        # Scrap shit.
        data = data['data']['repository']['pullRequests']

        # Collect pull numbers.
        for item in data['nodes']:
            numbers.append(item['number'])

        # Stop if no more pages.
        if not data['pageInfo']['hasNextPage']:
            break

        # Mover cursor.
        cursor = "\"" + data['pageInfo']['endCursor'] + "\""

    return numbers

def access_repositoriy_stargazer_count(owner, name):
    data = access('''{repository(owner:"$owner", name:"$name") {
                    	stargazerCount,
                       }}
                  
                  '''.replace("$owner", owner)
                        .replace("$name", name))
    
    return data['data']['repository']['stargazerCount']

def download_pulls_to_file(numbers, path):
    with open(path, 'w') as file:
        # Download this stuff.
        i = 0
        for number in numbers:
            i = i + 1
            # Some logging.
            if (i % 50 == 0):
                print("Downloading pull request " + str(i) + " from " + repo + " (" + path + ")")

            data = access("""
                    {repository(owner:"$owner", name:"$name") {
                    pullRequest(number:$number) {
                        id
                        number
                        title
                        body
                        createdAt
                        mergeCommit { oid }
                        closedAt
                        author { login }
                        reviewThreads(first: 100) {
                          edges {
                            node {
                              comments(first: 10) {
                                nodes {
                                  __typename
                                  ... on PullRequestReviewComment{
                                    path
                                    originalCommit {oid}
                                    commit{oid}
                                    url
                                    body
                                    diffHunk
                                    createdAt
                                    author {login}
                                    line
                                    startLine
                                    originalLine
                                    originalStartLine
                                }
                              }
                            }
                       	 	}
                          }
                          pageInfo {
                              hasNextPage
                          }
                        }
                    }
                }
}
                """.replace("$owner", owner)
                            .replace("$name", name)
                            .replace("$number", str(number)))

            if data is None:
                continue

            # Remove trash.
            data = data["data"]["repository"]

            # Write to file single line to file.
            file.write(json.dumps(data) + "\n")

# ----------------------------------------------------
# Clean wd.
# ----------------------------------------------------

print("Cleaning working directory")
os.system("rm -f *.zip")
os.system("rm -f *.json")

# ----------------------------------------------------
# Get random repository.
# ----------------------------------------------------

repo = access_random_repository()

owner = repo.split("/")[0]
name = repo.split("/")[1]

bucket = "vua-data"
location =  "raw-v2/" + repo + "/data.zip"

print("random repository: " + repo)

# ----------------------------------------------------
# Get Data From S3 if exists.
# ----------------------------------------------------
try:
    boto3.client('s3').download_file(bucket, location, "data.zip")
    print("Data exists in s3, updating")    
    os.system("unzip data.zip")    
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    
os.system("rm -f *.zip")
os.system("find . -name 'pulls*' -exec rm {} \;")

# ----------------------------------------------------
# Downloading pulls.
# ----------------------------------------------------

numbers = access_repository_pull_numbers(repo)

print("downloading " + str(len(numbers)) + " pulls")

if len(numbers) < 10:
    print("Less than 10 pulls, skip.")
    exit()

# split numbers into chunks
n_chunks = 6
chunks = np.array_split(numbers, n_chunks)

with Pool(n_chunks) as p:
    for idx, chunk in enumerate(chunks):
        p.apply_async(download_pulls_to_file, (chunk, "pulls" + str(idx) + ".json",))

    p.close()
    p.join()

# ----------------------------------------------------
# Get meta data about the repo and append.
# ----------------------------------------------------
with open("meta.json", 'a') as file:

    stargazers = access_repositoriy_stargazer_count(owner, name)

    file.write(json.dumps( {
        "repo": repo,
        "num_pulls": len(numbers),
        "num_stargazers": stargazers,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }) + "\n")

# ----------------------------------------------------
# Zipping and uploading
# ----------------------------------------------------

print("Zipping")
os.system("zip data.zip *.json")

if os.path.exists("data.zip"):
    print("Uploading to s3")
    boto3.client('s3').upload_file("data.zip", bucket, location)
else:
    print("zip failed")
