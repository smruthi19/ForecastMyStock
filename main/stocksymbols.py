import xlrd

path="stocksymbols3.xlsx"
inputWorkbook=xlrd.open_workbook(path)
inputWorksheet=inputWorkbook.sheet_by_index(0)
stocks=[]
print(inputWorksheet.nrows)
print(inputWorksheet.ncols)

print(inputWorksheet.cell_value(1,0))
for y in range(1, inputWorksheet.nrows):
    stocks.append(inputWorksheet.cell_value(y,0))

print(stocks)
