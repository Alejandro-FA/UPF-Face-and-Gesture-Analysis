for d in data/images_cropped_merged/*; do                   
    if [ -d "$d" ]; then
        python src/find_duplicates.py -d "$d"
    fi
done