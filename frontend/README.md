# Avatar Renderer MCP — Frontend

A professional, production-ready Next.js frontend for the Avatar Renderer MCP service. This provides a beautiful, interactive UI for generating talking avatar videos with a complete "choose → speak → generate → embed" workflow.

## Features

- 🎨 **Professional UI** with futuristic design and smooth animations
- 📤 **File Upload** for avatar images and audio files
- 🎭 **Avatar Gallery** with pre-built avatar options
- 📊 **Real-time Progress** tracking with visual feedback
- 🎬 **Video Preview** with autoplay and download options
- 📋 **Copy-to-Clipboard** embed code snippets
- 📱 **Fully Responsive** design for all screen sizes
- ⚡ **Optimized for Vercel** deployment

## Getting Started

### Prerequisites

- Node.js 18.x or later
- npm or yarn
- A running Avatar Renderer MCP backend (see parent directory)

### Installation

1. Install dependencies:

```bash
npm install
# or
yarn install
```

2. Configure environment variables:

```bash
cp .env.example .env.local
```

Edit `.env.local` and set your backend URL:

```env
NEXT_PUBLIC_AVATAR_API_BASE=http://localhost:8000
```

For production, use your deployed backend URL:

```env
NEXT_PUBLIC_AVATAR_API_BASE=https://your-backend-domain.com
```

### Development

Run the development server:

```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
npm run build
npm start
# or
yarn build
yarn start
```

## Deployment to Vercel

### Quick Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/avatar-renderer-mcp/tree/main/frontend)

### Manual Deployment

1. Install the Vercel CLI:

```bash
npm i -g vercel
```

2. Deploy:

```bash
vercel
```

3. Set environment variables in Vercel dashboard:

- Go to your project settings
- Add `NEXT_PUBLIC_AVATAR_API_BASE` with your backend URL
- Redeploy

### Important Note on Backend

⚠️ **The backend CANNOT run on Vercel** because it requires:
- GPU for model inference
- Large model files (several GB)
- Long-running processes (video rendering)
- FFmpeg and other system dependencies

Deploy the backend to:
- AWS EC2 / GCP Compute / Azure VM with GPU
- RunPod / Lambda Labs / Paperspace
- Kubernetes cluster (Helm charts provided in parent directory)

## API Integration

The frontend expects the backend to provide:

### `POST /render-upload`

Accepts multipart form data:
- `avatar`: Image file (PNG/JPG)
- `audio`: Audio file (WAV/MP3)
- `qualityMode`: String ("auto" | "real_time" | "high_quality")

Returns:
```json
{
  "jobId": "uuid",
  "statusUrl": "/status/uuid",
  "async": true
}
```

### `GET /status/{jobId}`

Returns either:
- JSON: `{"state": "processing" | "finished" | "error"}`
- Video: MP4 file (Content-Type: video/mp4)

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Main page with all UI
│   └── globals.css         # Global styles + Tailwind
├── public/                 # Static assets
├── package.json            # Dependencies
├── next.config.js          # Next.js config
├── tailwind.config.js      # Tailwind config
├── tsconfig.json           # TypeScript config
└── .env.local              # Local environment variables
```

## Customization

### Changing the Backend URL

Edit `.env.local`:

```env
NEXT_PUBLIC_AVATAR_API_BASE=https://your-backend.com
```

### Adding Custom Avatars

In `app/page.tsx`, update the `AVATARS` array:

```typescript
const AVATARS = [
  {
    id: 'your-avatar',
    name: 'Your Avatar',
    desc: 'Description',
    img: 'https://your-image-url.com/avatar.jpg'
  },
  // ...
];
```

### Styling

The app uses Tailwind CSS with custom utilities defined in `app/globals.css`.

Key utility classes:
- `.glass` - Glassmorphism effect
- `.neumorphic-dark` - Neumorphic card style
- `.glow-effect` - Glowing hover effect
- `.avatar-tile` - Avatar gallery tiles
- `.step-dot` - Progress step indicators

## Features Walkthrough

### 1. Hero Section
- Eye-catching title with gradient text
- Call-to-action buttons
- Feature badges
- Preview mockup

### 2. Interactive Wizard
- 4-step progress indicator
- Avatar selection gallery
- File upload for custom images/audio
- Script text area with sample options
- Quality mode selector
- Real-time progress tracking

### 3. Video Preview & Embed
- Auto-playing video player
- Download MP4 button
- Copy embed code snippet
- Syntax-highlighted code display

### 4. Use Cases Section
- Professional use case cards
- Icon-based visual hierarchy
- Gradient accents

### 5. Developer Integration
- Code examples for backend integration
- Terminal-styled code blocks
- Clear implementation guide

## Troubleshooting

### CORS Errors

Make sure your backend has CORS configured to allow requests from your frontend domain:

```python
# In backend app/api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-vercel-domain.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### File Upload Fails

Check that:
1. Backend `/render-upload` endpoint is implemented
2. Files are within size limits
3. File types are accepted (PNG/JPG for images, WAV/MP3 for audio)

### Video Doesn't Play

- Ensure the backend returns proper Content-Type: `video/mp4`
- Check browser console for errors
- Verify video file is valid MP4

## Browser Support

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (with fallbacks for some CSS features)
- Mobile browsers: Optimized for touch interactions

## Performance

- Optimized bundle size with Next.js automatic code splitting
- Images lazy-loaded
- Smooth animations with CSS transforms
- Efficient polling with cleanup

## License

See parent directory for license information.

## Contributing

See parent directory for contribution guidelines.

## Support

For issues and questions:
- Backend issues: See parent directory README
- Frontend issues: Create an issue in the repository
