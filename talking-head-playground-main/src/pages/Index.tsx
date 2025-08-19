import { useState } from "react";
import { PreviewStage } from "@/components/PreviewStage";
import { PipelineFlow } from "@/components/PipelineFlow";
import { RadialMenu } from "@/components/RadialMenu";
import { FilmstripDock } from "@/components/FilmstripDock";
import { FirstTimeOverlay } from "@/components/FirstTimeOverlay";
import { Button } from "@/components/ui/button";
import { PictureInPicture2, Sparkles, Settings } from "lucide-react";

const Index = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [hasImage, setHasImage] = useState(false);
  const [hasVideo, setHasVideo] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showFirstTime, setShowFirstTime] = useState(true);
  const [filmstripItems, setFilmstripItems] = useState([
    { id: '1', title: 'Take 1', status: 'ready' as const },
    { id: '2', title: 'Take 2', status: 'processing' as const },
  ]);

  const handleUploadImage = () => {
    setHasImage(true);
    setCurrentStep(Math.max(currentStep, 1));
  };

  const handleUploadVoice = () => {
    setCurrentStep(Math.max(currentStep, 2));
  };

  const handleEnterText = () => {
    setCurrentStep(Math.max(currentStep, 3));
  };

  const handleGenerate = () => {
    setIsGenerating(true);
    setCurrentStep(4);
    
    // Simulate generation process
    setTimeout(() => {
      setIsGenerating(false);
      setHasVideo(true);
      setFilmstripItems(prev => [
        ...prev,
        { id: Date.now().toString(), title: `Take ${prev.length + 1}`, status: 'ready' }
      ]);
    }, 3000);
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
    <div className="min-h-screen p-6">
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Talking Head Studio</h1>
          <p className="text-muted-foreground">Transform images into speaking videos with AI</p>
        </div>
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" className="glass-card">
            <PictureInPicture2 size={16} className="mr-2" />
            Preview Dock
          </Button>
          <Button variant="ghost" size="sm" className="glass-card">
            <Settings size={16} />
          </Button>
        </div>
      </header>

      {/* Main Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 max-w-7xl mx-auto">
        
        {/* Left Sidebar - Pipeline */}
        <div className="xl:col-span-1 space-y-6">
          <PipelineFlow currentStep={currentStep} onStepClick={handleStepClick} />
          
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

        {/* Center - Preview Stage */}
        <div className="xl:col-span-2">
          <PreviewStage
            hasImage={hasImage}
            hasVideo={hasVideo}
            isGenerating={isGenerating}
            onUploadImage={handleUploadImage}
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
            <p className="text-xs text-muted-foreground">
              {hasImage ? 'Ready to generate!' : 'Upload an image first'}
            </p>
          </div>

          {/* Video Settings */}
          <div className="glass-card p-4 space-y-4">
            <h3 className="font-semibold text-sm">Video Settings</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-muted-foreground">Quality</label>
                <select className="w-full mt-1 bg-muted/20 border border-border rounded-lg px-3 py-2 text-sm">
                  <option>HD (1080p)</option>
                  <option>4K (2160p)</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground">Duration</label>
                <select className="w-full mt-1 bg-muted/20 border border-border rounded-lg px-3 py-2 text-sm">
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
          onPlayItem={(id) => console.log('Play:', id)}
          onDownloadItem={(id) => console.log('Download:', id)}
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
            <span className="text-xs">Video Preview</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default Index;
