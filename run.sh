
# For ImageMagick v6
convert -clear cache

# Create 'frames' directory if it doesn't exist
mkdir -p frames

# Remove contents of 'frames' directory
rm frames/*
# rm out*
python3 main.py input_image.jpg

# For ImageMagick v6
# convert -clear cache
# ./gif.sh
