---
name: refresh-youtube-cookies
description: Extract YouTube cookies from Chrome browser and update the YOUTUBE_COOKIES_B64 environment variable on Railway for the squashvid service. Use when YouTube video downloads are failing due to authentication issues.
tools: Bash
---

# Refresh YouTube Cookies

This skill extracts fresh YouTube cookies from the local Chrome browser and updates them on Railway.

## Steps

1. Extract cookies from Chrome using yt-dlp:

```bash
yt-dlp --cookies-from-browser chrome --cookies /tmp/youtube_cookies_fresh.txt --skip-download "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>&1 | head -5
```

2. Filter for YouTube-specific cookies:

```bash
grep -E "youtube\.com|youtu\.be" /tmp/youtube_cookies_fresh.txt > /tmp/youtube_cookies_filtered.txt
echo "Extracted $(wc -l < /tmp/youtube_cookies_filtered.txt) YouTube cookies"
```

3. Base64 encode the cookies:

```bash
base64 -i /tmp/youtube_cookies_filtered.txt | tr -d '\n' > /tmp/youtube_cookies_b64.txt
echo "Encoded to $(wc -c < /tmp/youtube_cookies_b64.txt) bytes"
```

4. Update Railway environment variable:

```bash
cd /Users/adityashankarkini/code/squashvid/squashvid
railway variables set YOUTUBE_COOKIES_B64="$(cat /tmp/youtube_cookies_b64.txt)"
```

5. Verify the variable was set:

```bash
railway variables 2>&1 | grep -i youtube
```

6. Ask user if they want to trigger a redeploy:

```
The cookies have been updated. Would you like me to trigger a redeploy with `railway redeploy`?
```

## Cleanup

```bash
rm -f /tmp/youtube_cookies_fresh.txt /tmp/youtube_cookies_filtered.txt /tmp/youtube_cookies_b64.txt
```
