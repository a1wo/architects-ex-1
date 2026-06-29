while true; do
  STATUS=$(nebius ai job list | grep alon-wolf)
  echo "$STATUS"
  if echo "$STATUS" | grep -qi provisioning; then
    sleep 10
  else
    break
  fi
done
