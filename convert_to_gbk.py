#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Convert server_agent.py to GBK encoding

# Read UTF-8 encoded file
with open('/root/autodl-tmp/.autodl/iot/server_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Write GBK encoded file
with open('/root/autodl-tmp/.autodl/iot/server_agent.py', 'w', encoding='gbk') as f:
    f.write(content)

print("File successfully converted to GBK encoding")