# Extremely simple flask application, will display 'Hello World!' on the screen when you run it
# Access it by running it, then going to whatever port its running on (It'll say which port it's running on).
from sys import prefix
from flask import Flask, render_template
import boto3
app = Flask(__name__)

def get_folders(bucket, prefix=''):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    folders = []
    for result in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
        for p in result.get('CommonPrefixes', []):
            folder_prefix = p.get('Prefix')
            objects = s3.list_objects_v2(Bucket=bucket, Prefix=folder_prefix)
            last_modified = max(obj['LastModified'] for obj in objects.get('Contents', []))
            folder_prefix = folder_prefix.replace(prefix, '').replace('/', '')
            folders.append({'name':folder_prefix, 'date':last_modified, 'url':f'/folder/{folder_prefix}'})
    folders = sorted(folders, key=lambda x: x['date'], reverse=True)
    return folders

def get_most_recent_object(bucket, prefix=''):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = response.get('Contents', [])
    if not objects:
        return None
    latest = max(objects, key=lambda obj: obj['LastModified'])

    # Generate a presigned URL for the most recently modified object
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': latest['Key']},
        ExpiresIn=3600  # URL expires in 1 hour
    )
    latest['url'] = url


    return latest

@app.route('/')
def folders():
    bucket = 'pokemon-ml'
    prefix = 'screenshots/'
    folders = get_folders(bucket, prefix)

    return render_template('index.html', folders=folders)

@app.route('/folder/<folder_name>')
def folder_view(folder_name):
    bucket = 'pokemon-ml'
    prefix = f'screenshots/{folder_name}/'
    latest = get_most_recent_object(bucket, prefix)
    image_datetime = latest['LastModified']
    image_url = latest['url']
    return render_template('folder.html', image_datetime=image_datetime, image_url=image_url)




if __name__ == '__main__':
    app.run()