#!/bin/bash

folder_name="/tmp/CLASP/"

# Check if the folder exists
if [ -d "$folder_name" ]; then
    # Delete the folder and its contents
    rm -r "$folder_name"
    echo "Folder '$folder_name' deleted."
else
    echo "Folder '$folder_name' does not exist."
fi