import { X, Sparkles, Image as ImageIcon, Mic, Type } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FirstTimeOverlayProps {
  onClose: () => void;
}

export const FirstTimeOverlay = ({ onClose }: FirstTimeOverlayProps) => {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="glass-card p-8 max-w-2xl w-full relative">
        <Button 
          variant="ghost" 
          size="sm" 
          className="absolute top-4 right-4"
          onClick={onClose}
        >
          <X size={20} />
        </Button>
        
        <div className="text-center space-y-6">
          <div className="w-20 h-20 mx-auto glass-card rounded-full flex items-center justify-center animate-float">
            <Sparkles size={40} className="text-primary" />
          </div>
          
          <div>
            <h2 className="text-3xl font-bold mb-4 gradient-text">
              Welcome to Talking Head Studio!
            </h2>
            <p className="text-muted-foreground text-lg">
              Transform any image into a speaking video with AI-powered technology
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
            <div className="text-center space-y-3">
              <div className="w-16 h-16 mx-auto glass-card rounded-full flex items-center justify-center">
                <ImageIcon size={32} className="text-primary" />
              </div>
              <h3 className="font-semibold">Upload Image</h3>
              <p className="text-sm text-muted-foreground">
                Choose a photo to bring to life
              </p>
            </div>
            
            <div className="text-center space-y-3">
              <div className="w-16 h-16 mx-auto glass-card rounded-full flex items-center justify-center">
                <Mic size={32} className="text-primary" />
              </div>
              <h3 className="font-semibold">Add Voice</h3>
              <p className="text-sm text-muted-foreground">
                Upload audio or use text-to-speech
              </p>
            </div>
            
            <div className="text-center space-y-3">
              <div className="w-16 h-16 mx-auto glass-card rounded-full flex items-center justify-center">
                <Type size={32} className="text-primary" />
              </div>
              <h3 className="font-semibold">Generate</h3>
              <p className="text-sm text-muted-foreground">
                Create your talking head video
              </p>
            </div>
          </div>
          
          <div className="pt-6">
            <Button 
              onClick={onClose}
              className="glass-button text-white px-8"
              size="lg"
            >
              Get Started
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

