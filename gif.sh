mkdir frames

# Check if frames were generated successfully
if [ -f frames/out000.png ]; then
    # Use convert to create a GIF from the frames
    convert -loop 0 -delay 20 frames/*.png -delay 200 output.png output.gif
    echo "GIF created successfully."
else
    echo "No frames found. Please check the 'frames' directory."
fi


