param(
    [string]$src,
    [string]$o
)

cl /LD /EHsc `
  /I"C:\Users\sudha\AppData\Local\Programs\Python\Python314\Include" `
  /I"C:\Users\sudha\Documents\2048_RL\.venv\Lib\site-packages\pybind11\include" `
  "$src" `
  /link `
  /LIBPATH:"C:\Users\sudha\AppData\Local\Programs\Python\Python314\libs" `
  /OUT:"$o"