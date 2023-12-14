'''
This file contains some general and useful tools, except for the functions and algorithms used in the modules.
'''


import numpy as np
import pymysql
import base64
import urllib.parse


def generate_download_link(file, filename):

    # check the file type
    file_type = filename.split('.')[-1]
    download_string = file_type.upper() if 'baseline_' not in filename else 'baseline'
    quoted_filename = urllib.parse.quote(filename)
    if file_type == 'zip':
        file_content = file.getvalue()
        encoded = base64.b64encode(file_content).decode()
    else:
        encoded = base64.b64encode(file).decode()
    href = f'<a href="data:application/{file_type};base64, {encoded}" download="{quoted_filename}">Download {download_string} File</a>'
    return href


def exec_mysql(sql):

    # Define the database connection parameters
    db_config = {
    "host": "10.26.50.228",  # Use Docker container hostname or IP address if needed
    "user": "root",
    "password": "123456",
    "db": "ramancloud_database",  # Use your database name
    "port": 3306,  # This should match the port mapping you used when running the container
    }

    # Create a connection to the database
    try:
        connection = pymysql.connect(**db_config)
        if connection.open:
            cursor = connection.cursor()
            cursor.execute(sql)
        connection.commit()

    except pymysql.Error as e:
        print(f"Error: {e}")
    finally:
        # Close the cursor and database connection
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals() and connection.open:
            connection.close()
