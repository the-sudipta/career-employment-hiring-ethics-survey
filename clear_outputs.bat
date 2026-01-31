@echo off
setlocal

:: "outputs" ফোল্ডারটির পাথ সেট করা হচ্ছে (বর্তমান লোকেশনে)
set "targetFolder=%~dp0outputs"

:: চেক করা হচ্ছে ফোল্ডারটি আছে কিনা
if exist "%targetFolder%" (
    echo 'outputs' folder found. Cleaning contents...
    
    :: সব ফাইল ডিলিট করার কমান্ড
    del /f /q "%targetFolder%\*"
    
    :: সব সাব-ফোল্ডার ডিলিট করার কমান্ড
    for /d %%x in ("%targetFolder%\*") do @rd /s /q "%%x"
    
    echo Done!
) else (
    :: ফোল্ডার না থাকলে চুপচাপ এক্সিট করবে
    goto :eof
)

endlocal