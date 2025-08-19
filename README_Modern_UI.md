# Talking Head Studio - Modern UI

A modern, beautiful React-based user interface for the Talking Head Studio, inspired by the talking-head-playground design. This UI provides a sleek, glass-morphism design with smooth animations and an intuitive user experience.

## Features

- 🎨 **Modern Glass-Morphism Design**: Beautiful glass cards with backdrop blur effects
- 🎯 **Intuitive Pipeline Flow**: Step-by-step creation process with visual progress indicators
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- ⚡ **Real-time Feedback**: Toast notifications and loading states
- 🎬 **Video Management**: Filmstrip dock for managing generated videos
- 🎵 **Multiple Input Methods**: Support for image upload, audio upload, and text-to-speech
- 🔄 **Seamless Integration**: Connects to your existing SadTalker backend

## Project Structure

```
talking-head/
├── src/                          # React application source
│   ├── components/               # UI components
│   │   ├── ui/                   # Base UI components (buttons, toasts, etc.)
│   │   ├── PreviewStage.tsx      # Main preview area
│   │   ├── PipelineFlow.tsx      # Creation pipeline
│   │   ├── FilmstripDock.tsx     # Video management
│   │   └── ...
│   ├── pages/                    # Page components
│   ├── lib/                      # Utility functions
│   └── index.css                 # Design system and styles
├── api_server.py                 # FastAPI backend server
├── package.json                  # Node.js dependencies
├── tailwind.config.ts           # Tailwind CSS configuration
└── vite.config.ts               # Vite build configuration
```

## Setup Instructions

### Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.8+ with your existing SadTalker environment
- FastAPI and uvicorn for the API server

### 1. Install Frontend Dependencies

```bash
# Install Node.js dependencies
npm install

# Or using yarn
yarn install
```

### 2. Install Backend Dependencies

```bash
# Install FastAPI and uvicorn
pip install fastapi uvicorn python-multipart

# Make sure your existing SadTalker environment is set up
# (Follow your existing setup instructions)
```

### 3. Start the Backend Server

```bash
# Start the FastAPI server (runs on port 7860)
python api_server.py
```

### 4. Start the Frontend Development Server

```bash
# Start the React development server (runs on port 3000)
npm run dev

# Or using yarn
yarn dev
```

### 5. Access the Application

Open your browser and navigate to:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:7860

## Usage

1. **Upload Image**: Click the upload button or drag an image to the preview area
2. **Add Voice**: Upload an audio file or enter text for text-to-speech
3. **Generate**: Click the generate button to create your talking head video
4. **Manage**: View and download your generated videos in the filmstrip dock

## Design System

The UI uses a custom design system with:

- **Colors**: Purple/blue primary theme with teal accents
- **Glass Effects**: Backdrop blur with transparency
- **Animations**: Smooth transitions and floating effects
- **Typography**: Modern, clean font hierarchy
- **Spacing**: Consistent 8px grid system

## API Integration

The frontend communicates with your SadTalker backend through:

- **POST /api/generate**: Generate talking head videos
- **GET /api/health**: Health check endpoint
- **GET /api/models**: Get available models and settings

## Development

### Building for Production

```bash
# Build the React application
npm run build

# The built files will be in the `dist/` directory
```

### Customization

- **Colors**: Modify CSS variables in `src/index.css`
- **Components**: Edit components in `src/components/`
- **API**: Modify endpoints in `api_server.py`

## Troubleshooting

### Common Issues

1. **CORS Errors**: Make sure the backend server is running on port 7860
2. **Module Not Found**: Run `npm install` to install dependencies
3. **API Connection**: Check that the FastAPI server is running and accessible

### Debug Mode

```bash
# Start frontend in development mode with debugging
npm run dev

# Start backend with detailed logging
python api_server.py --log-level debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project follows the same license as your existing SadTalker implementation.

---

**Note**: This modern UI is designed to work seamlessly with your existing SadTalker backend. Make sure all your existing dependencies and models are properly set up before running the new interface.

