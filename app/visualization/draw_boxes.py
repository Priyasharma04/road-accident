SAFE=(0,255,0)
WARNING=(0,255,255)
DANGER=(0,0,255)
if status=="danger":
    color=DANGER

elif status=="warning":
    color=WARNING

else:
    color=SAFE