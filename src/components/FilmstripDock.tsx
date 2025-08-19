import { Play, Download, MoreVertical } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FilmstripItem {
  id: string;
  title: string;
  status: 'ready' | 'processing' | 'error';
  videoUrl?: string;
  thumbnail?: string;
}

interface FilmstripDockProps {
  items: FilmstripItem[];
  onPlayItem: (id: string) => void;
  onDownloadItem: (id: string) => void;
}

export const FilmstripDock = ({ items, onPlayItem, onDownloadItem }: FilmstripDockProps) => {
  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-sm">Generated Videos</h3>
        <Button variant="ghost" size="sm">
          <MoreVertical size={16} />
        </Button>
      </div>
      
      <div className="flex gap-4 overflow-x-auto pb-2">
        {items.map((item) => (
          <div key={item.id} className={`filmstrip-item ${item.status === 'processing' ? 'processing' : ''}`}>
            {item.thumbnail ? (
              <img 
                src={item.thumbnail} 
                alt={item.title}
                className="w-full h-full object-cover rounded-lg"
              />
            ) : (
              <div className="w-full h-full bg-muted/50 rounded-lg flex items-center justify-center">
                <span className="text-xs text-muted-foreground">{item.title}</span>
              </div>
            )}
            
            {/* Overlay with actions */}
            <div className="absolute inset-0 bg-black/50 opacity-0 hover:opacity-100 transition-opacity duration-200 rounded-lg flex items-center justify-center gap-2">
              <Button 
                size="sm" 
                variant="secondary"
                onClick={() => onPlayItem(item.id)}
                className="w-8 h-8 p-0"
              >
                <Play size={12} />
              </Button>
              <Button 
                size="sm" 
                variant="secondary"
                onClick={() => onDownloadItem(item.id)}
                className="w-8 h-8 p-0"
              >
                <Download size={12} />
              </Button>
            </div>
            
            {/* Status indicator */}
            <div className="absolute top-1 right-1">
              <div className={`w-2 h-2 rounded-full ${
                item.status === 'ready' ? 'bg-green-500' :
                item.status === 'processing' ? 'bg-yellow-500' :
                'bg-red-500'
              }`} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

