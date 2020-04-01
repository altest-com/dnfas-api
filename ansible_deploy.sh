#!/bin/bash
ansible-playbook ./ansible/deploy.yml -K -u $USER -i ./ansible/hosts -vvv