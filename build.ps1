param(
    [string]$Source,
    [string]$Output
)

cl /LD /EHsc `
  /I"C:\Users\sudha\AppData\Local\Programs\Python\Python314\Include" `
  /I"C:\Users\sudha\Documents\2048_RL\.venv\Lib\site-packages\pybind11\include" `
  "$Source.cpp" `
  /link `
  /LIBPATH:"C:\Users\sudha\AppData\Local\Programs\Python\Python314\libs" `
  /OUT:"$Output.pyd"