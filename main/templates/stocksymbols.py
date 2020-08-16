import xlrd

path="Stock.xlsx"
inputWorkbook=xlrd.open_workbook(path)
inputWorksheet=inputWorkbook.sheet_by_index(0)

print(inputWorksheet.nrows)
print(inputWorksheet.ncols)
