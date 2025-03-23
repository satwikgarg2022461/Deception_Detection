@echo off
echo Running Deception Detection Models...

set TASK=sender
set POWER=

if "%1"=="receiver" set TASK=receiver
if "%2"=="power" set POWER=--power

echo.
echo Running Baselines...
python implement_baselines.py

echo.
echo Running Harbingers Model with task=%TASK% power=%POWER%...
python implement_harbingers.py --task %TASK% %POWER%

echo.
echo Running Bag of Words Model with task=%TASK% power=%POWER%...
python implement_bagofwords.py --task %TASK% %POWER%

echo.
echo Running LSTM Model with task=%TASK% power=%POWER%...
python implement_lstm.py --task %TASK% %POWER%

echo.
echo Running ContextLSTM Model with task=%TASK% power=%POWER%...
python implement_contextlstm.py --task %TASK% %POWER%

echo.
echo Running BERT+Context Model with task=%TASK% power=%POWER%...
python implement_bertcontext.py --task %TASK% %POWER%

echo.
echo All models completed!
