from osgeo import gdal, osr
import sys


def convert_tiff_to_epsg4326(input_tiff, output_tiff):
    # Open the input TIFF file
    src_ds = gdal.Open(input_tiff)

    # Get the source CRS
    src_crs = src_ds.GetProjection()

    # Define the target CRS as EPSG:4326
    target_crs = osr.SpatialReference()
    target_crs.ImportFromEPSG(4326)

    # Create a transformer to convert from the source CRS to EPSG:4326
    transformer = osr.CoordinateTransformation(osr.SpatialReference(src_crs), target_crs)

    # Get the raster band from the source dataset
    band = src_ds.GetRasterBand(1)

    # Get the geotransform (transform) of the source dataset
    transform = src_ds.GetGeoTransform()

    # Get the size (width and height) of the source dataset
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize

    # Create a new output raster dataset with the target CRS
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_tiff, width, height, 1, band.DataType)

    # Apply the coordinate transformation and write the data to the new dataset
    gdal.ReprojectImage(src_ds, dst_ds, src_crs, target_crs.ExportToWkt(), gdal.GRA_NearestNeighbour)

    # Set the geotransform for the new dataset
    dst_ds.SetGeoTransform(transform)

    # Set the CRS for the new dataset
    dst_ds.SetProjection(target_crs.ExportToWkt())

    # Close the datasets
    src_ds = None
    dst_ds = None

	
	
input_tiff_file = sys.argv[-2].lower()
output_tiff_file = sys.argv[-1].lower()

convert_tiff_to_epsg4326(input_tiff_file, output_tiff_file)
