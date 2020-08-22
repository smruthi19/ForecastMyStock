import xlrd

path="StockSymbol.xlsx"
inputWorkbook=xlrd.open_workbook(path)
inputWorksheet=inputWorkbook.sheet_by_index(0)
stocks=[]
print(inputWorksheet.nrows)
print(inputWorksheet.ncols)

for y in range(1, inputWorksheet.nrows):
    stocks.append(inputWorsheet.cell_value(y,1))

print(stocks)
