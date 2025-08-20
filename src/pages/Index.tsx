import { useState, useRef, useEffect } from "react";
import { PreviewStage } from "@/components/PreviewStage";
import { PipelineFlow } from "@/components/PipelineFlow";
import { RadialMenu } from "@/components/RadialMenu";
import { FilmstripDock } from "@/components/FilmstripDock";
import { FirstTimeOverlay } from "@/components/FirstTimeOverlay";
import { Button } from "@/components/ui/button";
import { PictureInPicture2, Sparkles, Settings, Moon, Sun } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import axios from "axios";

interface FilmstripItem {
  id: string;
  title: string;
  status: 'ready' | 'processing' | 'error';
  videoUrl?: string;
  thumbnail?: string;
}

const Index = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [hasImage, setHasImage] = useState(false);
  const [hasVideo, setHasVideo] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showFirstTime, setShowFirstTime] = useState(true);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [selectedAudio, setSelectedAudio] = useState<File | null>(null);
  const [textInput, setTextInput] = useState("");
  const [filmstripItems, setFilmstripItems] = useState<FilmstripItem[]>([]);
  const [currentPlayingVideo, setCurrentPlayingVideo] = useState<string | null>(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [liveLogs, setLiveLogs] = useState<string[]>([]);
  
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  // Check for saved theme preference on component mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
      setIsDarkMode(true);
      document.documentElement.classList.add('dark');
    }
  }, []);

  // Toggle dark mode
  const toggleDarkMode = () => {
    const newDarkMode = !isDarkMode;
    setIsDarkMode(newDarkMode);
    
    if (newDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  };

  const handleUploadImage = () => {
    fileInputRef.current?.click();
  };

  const handleImageFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setHasImage(true);
      setCurrentStep(Math.max(currentStep, 1));
      toast({
        title: "Image uploaded successfully!",
        description: "Your image is ready for processing.",
      });
    }
  };

  const handleUploadVoice = () => {
    audioInputRef.current?.click();
  };

  const handleAudioFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedAudio(file);
      setCurrentStep(Math.max(currentStep, 2));
      toast({
        title: "Audio uploaded successfully!",
        description: "Your audio is ready for processing.",
      });
    }
  };

  const handleEnterText = () => {
    if (textInput.trim()) {
      setCurrentStep(Math.max(currentStep, 3));
      toast({
        title: "Text added successfully!",
        description: "Your script is ready for generation.",
      });
    } else {
      toast({
        title: "Please enter some text",
        description: "Add a script for the talking head to speak.",
        variant: "destructive",
      });
    }
  };

  const handleGenerate = async () => {
    if (!selectedImage) {
      toast({
        title: "No image selected",
        description: "Please upload an image first.",
        variant: "destructive",
      });
      return;
    }

    setIsGenerating(true);
    setLiveLogs([]);
    setCurrentStep(4);
    setCurrentPlayingVideo(null); // Clear any currently playing video
    
    let es: EventSource | null = null;
    try {
      // Start listening to live logs via SSE
      const dev = import.meta && (import.meta as any).env && (import.meta as any).env.DEV;
      const apiPort = ((import.meta as any).env && (import.meta as any).env.VITE_API_PORT) || 7860;
      const logsUrl = dev ? `http://localhost:${apiPort}/api/logs/stream` : '/logs/stream';
      es = new EventSource(logsUrl);
      es.onopen = () => {
        setLiveLogs(prev => [...prev, '[logs] connected']);
      };
      es.onmessage = (e) => {
        const msg = (e.data || '').toString();
        // Filter warnings just in case
        if (!msg.startsWith('WARNING:') && !msg.toLowerCase().includes('safetensor')) {
          setLiveLogs(prev => [...prev, msg]);
        }
      };
      es.onerror = () => {
        // Let browser auto-retry; surface a single line once
        setLiveLogs(prev => prev.length && prev[prev.length-1].includes('reconnecting') ? prev : [...prev, '[logs] reconnecting...']);
      };
      // Create FormData for the API call
      const formData = new FormData();
      formData.append('image', selectedImage);
      
      if (selectedAudio) {
        formData.append('audio', selectedAudio);
      }
      
      if (textInput.trim()) {
        formData.append('text', textInput);
      }

      // Call the SadTalker API
      const response = await axios.post('/generate', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob',
      });

      // Create video URL from response
      const videoUrl = URL.createObjectURL(response.data);
      
      // Add to filmstrip
      const newItem: FilmstripItem = {
        id: Date.now().toString(),
        title: `Take ${filmstripItems.length + 1}`,
        status: 'ready',
        videoUrl,
        thumbnail: URL.createObjectURL(selectedImage),
      };
      
      setFilmstripItems(prev => [...prev, newItem]);
      setHasVideo(true);
      
      toast({
        title: "Video generated successfully!",
        description: "Your talking head video is ready.",
      });
      
    } catch (error) {
      console.error('Generation error:', error);
      toast({
        title: "Generation failed",
        description: "There was an error generating your video. Please try again.",
        variant: "destructive",
      });
    } finally {
      if (es) {
        es.close();
      }
      setIsGenerating(false);
    }
  };

  // Video action handlers
  const handlePlayVideo = (id: string) => {
    const item = filmstripItems.find(item => item.id === id);
    if (item && item.videoUrl) {
      // If clicking the same video, stop playing it
      if (currentPlayingVideo === id) {
        setCurrentPlayingVideo(null);
        toast({
          title: "Video stopped",
          description: "Video playback has been stopped.",
        });
      } else {
        setCurrentPlayingVideo(id);
        toast({
          title: "Video selected",
          description: "Video is now displayed in the preview area.",
        });
      }
    }
  };

  const handleDownloadVideo = (id: string) => {
    const item = filmstripItems.find(item => item.id === id);
    if (item && item.videoUrl) {
      // Create a download link
      const link = document.createElement('a');
      link.href = item.videoUrl;
      link.download = `talking_head_${item.title.replace(/\s+/g, '_')}.mp4`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      toast({
        title: "Download started",
        description: "Your video download has begun.",
      });
    }
  };

  const handleStepClick = (step: number) => {
    if (step <= currentStep) {
      // Allow going back to completed steps
      if (step === 0) handleUploadImage();
      if (step === 1) handleUploadVoice();
      if (step === 2) handleEnterText();
      if (step === 3) handleGenerate();
    }
  };

  return (
    <div className={`min-h-screen p-6 transition-colors duration-300 ${isDarkMode ? 'dark bg-gray-900 text-white' : 'bg-gradient-to-br from-purple-50 to-white text-gray-900'}`}>
      {/* Hidden file inputs */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleImageFileSelect}
        className="hidden"
      />
      <input
        ref={audioInputRef}
        type="file"
        accept="audio/*"
        onChange={handleAudioFileSelect}
        className="hidden"
      />

      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Talking Head Studio</h1>
          <p className={`${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Transform images into speaking videos with AI</p>
        </div>
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" className="glass-card">
            <PictureInPicture2 size={16} className="mr-2" />
            Preview Dock
          </Button>
          <Button 
            variant="ghost" 
            size="sm" 
            className={`glass-card transition-all duration-300 ${isDarkMode ? 'hover:bg-gray-700/50' : 'hover:bg-gray-100/50'}`}
            onClick={toggleDarkMode}
            title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            {isDarkMode ? <Sun size={16} className="text-yellow-400" /> : <Moon size={16} className="text-blue-600" />}
          </Button>
          <Button variant="ghost" size="sm" className="glass-card">
            <Settings size={16} />
          </Button>
        </div>
      </header>

      {/* Main Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 max-w-7xl mx-auto px-4">
        
        {/* Left Sidebar - Quick Actions Only */}
        <div className="xl:col-span-1 space-y-6">
          {/* Quick Actions */}
          <div className="glass-card p-4 space-y-3">
            <h3 className="font-semibold text-sm">Quick Actions</h3>
            <div className="space-y-2">
              <Button 
                variant="ghost" 
                size="sm" 
                className="w-full justify-start text-left"
                onClick={handleUploadImage}
              >
                📸 Upload New Image
              </Button>
              <Button 
                variant="ghost" 
                size="sm" 
                className="w-full justify-start text-left"
                onClick={handleUploadVoice}
              >
                🎤 Record Voice
              </Button>
              <Button 
                variant="ghost" 
                size="sm" 
                className="w-full justify-start text-left"
                onClick={handleEnterText}
              >
                ✨ Text-to-Speech
              </Button>
            </div>
          </div>
        </div>

        {/* Center - Creation Pipeline + Preview Stage */}
        <div className="xl:col-span-2 space-y-6">
          {/* Creation Pipeline moved to top */}
          <div className="glass-card p-6">
            <PipelineFlow currentStep={currentStep} onStepClick={handleStepClick} />
          </div>
          
          {/* Preview Stage */}
          <PreviewStage
            hasImage={hasImage}
            hasVideo={hasVideo}
            isGenerating={isGenerating}
            onUploadImage={handleUploadImage}
            selectedImage={selectedImage}
            selectedAudio={selectedAudio}
            textInput={textInput}
            onTextChange={setTextInput}
            currentPlayingVideo={currentPlayingVideo}
            filmstripItems={filmstripItems}
            logs={liveLogs}
          />
        </div>

        {/* Right Sidebar - Controls */}
        <div className="xl:col-span-1 space-y-6">
          {/* Generate Button */}
          <div className="glass-card p-6 text-center">
            <Button
              onClick={handleGenerate}
              disabled={!hasImage || isGenerating}
              className="w-full glass-button text-white mb-4"
              size="lg"
            >
              <Sparkles className="mr-2" size={20} />
              {isGenerating ? 'Generating...' : 'Generate Video'}
            </Button>
            <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              {hasImage ? 'Ready to generate!' : 'Upload an image first'}
            </p>
          </div>

          {/* Video Settings */}
          <div className="glass-card p-4 space-y-4">
            <h3 className="font-semibold text-sm">Video Settings</h3>
            <div className="space-y-3">
              <div>
                <label className={`text-xs font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Quality</label>
                <select className={`w-full mt-1 bg-muted/20 border border-border rounded-lg px-3 py-2 text-sm transition-colors duration-200 ${
                  isDarkMode 
                    ? 'bg-gray-700/20 border-gray-600 text-white' 
                    : 'bg-white/50 border-gray-300 text-gray-900'
                }`}>
                  <option>HD (1080p)</option>
                  <option>4K (2160p)</option>
                </select>
              </div>
              <div>
                <label className={`text-xs font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Duration</label>
                <select className={`w-full mt-1 bg-muted/20 border border-border rounded-lg px-3 py-2 text-sm transition-colors duration-200 ${
                  isDarkMode 
                    ? 'bg-gray-700/20 border-gray-600 text-white' 
                    : 'bg-white/50 border-gray-300 text-gray-900'
                }`}>
                  <option>Auto</option>
                  <option>30 seconds</option>
                  <option>1 minute</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Filmstrip */}
      <div className="mt-8 max-w-7xl mx-auto">
        <FilmstripDock
          items={filmstripItems}
          onPlayItem={handlePlayVideo}
          onDownloadItem={handleDownloadVideo}
        />
      </div>

      {/* Floating Radial Menu */}
      <RadialMenu
        onUploadImage={handleUploadImage}
        onUploadVoice={handleUploadVoice}
        onEnterText={handleEnterText}
        onSettings={() => console.log('Settings')}
      />

      {/* First Time User Overlay */}
      {showFirstTime && (
        <FirstTimeOverlay onClose={() => setShowFirstTime(false)} />
      )}

      {/* Always-On Preview Dock (when not in focus) */}
      {hasVideo && (
        <div className="fixed bottom-4 left-4 w-48 aspect-video glass-card p-2 z-40">
          <div className="w-full h-full bg-gradient-to-br from-primary/20 to-secondary/20 rounded-lg flex items-center justify-center">
            <span className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Video Preview</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default Index;
