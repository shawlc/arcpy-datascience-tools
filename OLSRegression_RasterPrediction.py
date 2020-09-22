"""
THIS SCRIPT APPLIES OLS REGRESSION FROM SCRATCH USING NUMPY

To create an ArcToolbox tool with which to execute this script, do the following.
1   In  ArcMap > Catalog > Toolboxes > My Toolboxes, either select an existing toolbox
    or right-click on My Toolboxes and use New > Toolbox to create (then rename) a new one.
2   Drag (or use ArcToolbox > Add Toolbox to add) this toolbox to ArcToolbox.
3   Right-click on the toolbox in ArcToolbox, and use Add > Script to open a dialog box.
4   In this Add Script dialog box, use Label to name the tool being created, and press Next.
5   In a new dialog box, browse to the .py file to be invoked by this tool, and press Next.
6   In the next dialog box, specify the following inputs (using dropdown menus wherever possible)
    before pressing OK or Finish.

        DISPLAY NAME            DATA TYPE           PROPERTY>DIRECTION>VALUE    PROPERTY>DEFAULT>VALUE
        Indepedent Rasters       Raster Dataset      Input                       Multivalue
        Rasters to Predict      Raster Dataset      Input                       Multivalue
        Dependent Raster      Raster Dataset      Input                       Optional
        Output                  Raster Dataset      Output

The indepedent rasters are the rasters for which coefficients are calculated to predict the dependent raster.
The indepedent rasters must have the same extent as the dependent raster
The output will be the dot product of the coefficients and the rasters to predict. As such, the rasters to predict
must be of the same numerical range and interpretation as the indepedent variables. The rasters to predict must be
listed in the same order as the indepedent rasters.

Example:

    Indepedent Rasters = [Developed Area in City A, Population in City A]
    Rasters to Predict = [Developed Area in City B, Population in City B]
    Dependent Raster = Household Income in City A
    Output (to be calculated by tool) = Household Income in City B

7   To later revise any of this, right-click to the tool's name and select Properties.
"""

# Import external modules
import sys,string,os,arcpy,traceback
import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm
from arcpy.sa import *

arcpy.env.overwriteOutput=True

try:

        value_table=arcpy.ValueTable()
        result_table=arcpy.ValueTable()

        value_table.loadFromString(arcpy.GetParameterAsText(0))
        target=arcpy.GetParameterAsText(2)
        output = arcpy.GetParameterAsText(3)

        if arcpy.GetParameterAsText(1):
            result_table.loadFromString(arcpy.GetParameterAsText(1))
        else:
            result_table.loadFromString(arcpy.GetParameterAsText(0))

        def raster_to_grid(input_table):
            input_grid_name_list = []

            # Loop through the list of inputs
            for i in range(0,input_table.rowCount):
                input_grid_name_list.append(input_table.getRow(i))

            #rasterlist= map(lambda x:Con(IsNull(x),-999,x),input_grid_name_list)
            rasterlist=map(lambda x:x.replace("'", ""),input_grid_name_list)

            compgrid = arcpy.env.workspace+"\\"+"tempcomp"

            #COMBINE NODES INTO A SINGLE MULTIBAND IMAGE
            arcpy.CompositeBands_management(rasterlist,compgrid)
            arcpy.AddMessage("\nMultiband Image located in : "+compgrid)

            return compgrid
            #CONVERT THE MULTIBAND RASTER INTO AN ARRAY

        def raster_to_array(input_table):
            new_grid = raster_to_grid(input_table)
            out_array=arcpy.RasterToNumPyArray(Raster(new_grid))
            return out_array.astype(float)

        compArray = raster_to_array(value_table)
        numHeight=compArray.shape[0]
        numRow=compArray.shape[1]
        numCol=compArray.shape[2]

        #CONVERT THE DEPENDENT RASTER INTO AN ARRAY
        targetArray=arcpy.RasterToNumPyArray(target)
        targetArray=targetArray.astype(float)
        targetRow=targetArray.shape[0]  # an integer indicating number of rows in input and output grids
        targetCol=targetArray.shape[1]
        arcpy.AddMessage("\nTarget array has "+str(targetRow)+" rows and and "+str(targetCol)+" columns")

        arcpy.AddMessage("\nComposite array has "+str(numHeight)+" elements in " +
                         str(numRow)+" rows and and "+str(numCol)+" columns")

        if targetRow == numRow and targetCol == numCol:
            arcpy.AddMessage("\nTarget array and composite array have equal dimensions")

        indepedentArray=np.asarray(map(lambda x:x.flatten(),compArray))
        dependentarray=targetArray.flatten()

        arcpy.AddMessage("\nIndepedent array has " + str(len(indepedentArray[0]))+" length and is of type" + str(type(indepedentArray[0][0])))
        arcpy.AddMessage("\nDependent array has "+str(len(dependentarray))+" length and is of type"+str(type(dependentarray[0])))

        #Borrowed numpy least squares regression code from stackoverflow
        #https://stackoverflow.com/questions/11479064/multiple-linear-regression-in-python
        indepedentArray=indepedentArray.T  # transpose so input vectors are along the rows
        indepedentArray=np.c_[indepedentArray,np.ones(indepedentArray.shape[0])]  # add bias term
        beta_hat=np.linalg.lstsq(indepedentArray,dependentarray)[0]

        arcpy.AddMessage("\nBeta array length is "+str(len(beta_hat)))

        #My final output is a prediction created from the dot product of the beta hats with the list of result rasters
        resultArray=raster_to_array(result_table)
        resHeight=resultArray.shape[0]
        resRow=resultArray.shape[1]
        resCol=resultArray.shape[2]

        dotComp=np.transpose(resultArray,(1,2,0))

        outGrid=np.empty((resRow,resCol))
        for row in range(resRow):
            for col in range(resCol):
                    outGrid[row][col]=np.dot(np.append(dotComp[row][col],1),beta_hat)

        arcpy.AddMessage("\nOutput array has "+str(outGrid.shape[0])+" rows and "+
                         str(outGrid.shape[1])+" columns")

        raster_ref = result_table.getRow(0)[1:-1]

        arcpy.AddMessage("\nRaster reference is "+str(raster_ref))

        spatial = arcpy.Describe(raster_ref).spatialReference

        cell_size_x=str(arcpy.GetRasterProperties_management(raster_ref,"CELLSIZEX"))
        cell_size_y=str(arcpy.GetRasterProperties_management(raster_ref,"CELLSIZEY"))
        x_min=str(arcpy.GetRasterProperties_management(raster_ref,"LEFT"))
        y_min=str(arcpy.GetRasterProperties_management(raster_ref,"BOTTOM"))

        outRaster=arcpy.NumPyArrayToRaster(outGrid,
                                              x_cell_size=float(cell_size_x),
                                              y_cell_size=float(cell_size_y),
                                              lower_left_corner=arcpy.Point(x_min,y_min),
                                              value_to_nodata=0)

        arcpy.DefineProjection_management(outRaster, spatial)
        outRaster.save(output)

except Exception as e:
        # If unsuccessful, end gracefully by indicating why
        arcpy.AddError('\n'+"Script failed because: \t\t"+e.message)
        # ... and where
        exceptionreport=sys.exc_info()[2]
        fullermessage=traceback.format_tb(exceptionreport)[0]
        arcpy.AddError("at this location: \n\n"+fullermessage+"\n")

