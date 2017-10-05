#/bin/bash
ps -ef | grep "fceux" | grep -v grep | awk '{print "kill -9",$2}' | sh -v
