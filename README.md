# ImageAverage
In order to more efficiently and accurately analyze the RGB variation data of fixed-pixel regions in RAW-format images, we employed a Python-based RAW image batch-processing program (ImageAverage) to process the images. RGB color data was extracted from fixed regions of each image until all RAW images in the folder were processed. 
# Install the library
rawpy>=0.20.0
numpy>=1.20.0
Pillow>=9.0.0

# run the code (Windows 11)
Copy the provided code, paste it into the file, and save it. Ensure that the file is located in the root directory of drive C.
Directly double-click to open the "main.py" code, and a selection dialog box will appear. After selecting the folder containing RAW-format images, you will enter the coordinate selection dialog box. You can zoom in or out of the image using the mouse wheel or the buttons in the dialog box. Use the mouse to click and select the center or the brightest spot of the spot. Click "Confirm selection and Process" to start processing the average RGB values of the 2*2-pixel area at the selected coordinates in each image within the folder. Once the processing is complete, a CSV file will be automatically generated, and you can directly click "Open File Location" to access the data file. The data file mainly includes the image name, processing coordinates, the average, maximum, and minimum RGB values for the three channels in the pixel area. The "G" column contains the data for the G channel, which serves as the primary analytical data.
