import numpy as np
import os
import sys
import wave
import pyttsx3
import uuid
import time
import subprocess
from PIL import Image

try:
    import trimesh
except ImportError:
    print("Installing trimesh...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh"])
    import trimesh

class TexturedTalkingHead3D:
    def __init__(self, obj_path, texture_path=None):
        self.obj_path = obj_path
        self.texture_path = texture_path
        self.mesh = None
        self.texture_image = None
        self.frame_count = 0
        self.output_dir = None
        
        # Animation parameters
        self.lip_sync_intensity = 0.0
        self.expression_intensity = 0.0
        self.head_rotation = 0.0
        
        # Load the model
        self.load_model()
        
        # Load texture if available
        if texture_path and os.path.exists(texture_path):
            self.load_texture()
    
    def load_model(self):
        """Load the OBJ model"""
        try:
            print(f"Loading model: {self.obj_path}")
            self.mesh = trimesh.load(self.obj_path)
            print(f"Model loaded successfully!")
            print(f"Vertices: {len(self.mesh.vertices)}")
            print(f"Faces: {len(self.mesh.faces)}")
            
            # Store original vertices for animation
            self.original_vertices = self.mesh.vertices.copy()
            
            # Center and scale
            self.mesh.vertices -= self.mesh.vertices.mean(axis=0)
            max_dim = np.max(self.mesh.vertices.max(axis=0) - self.mesh.vertices.min(axis=0))
            scale = 1.0 / max_dim
            self.mesh.vertices *= scale
            
            # Rotate to get face facing camera: first flip upright, then make vertical
            # Step 1: Rotate 180 degrees around X-axis to flip face up
            cos_180 = np.cos(np.pi)
            sin_180 = np.sin(np.pi)
            for i in range(len(self.mesh.vertices)):
                x, y, z = self.mesh.vertices[i]
                # Rotate around X-axis to flip upright
                self.mesh.vertices[i] = [x, y * cos_180 - z * sin_180, y * sin_180 + z * cos_180]
            
            # Step 2: Rotate 90 degrees around Z-axis to make vertical
            cos_90 = np.cos(np.pi/2)
            sin_90 = np.sin(np.pi/2)
            for i in range(len(self.mesh.vertices)):
                x, y, z = self.mesh.vertices[i]
                # Rotate around Z-axis to make vertical
                self.mesh.vertices[i] = [x * cos_90 - y * sin_90, x * sin_90 + y * cos_90, z]
            
            self.original_vertices = self.mesh.vertices.copy()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def load_texture(self):
        """Load texture from image file"""
        try:
            print(f"Loading texture: {self.texture_path}")
            self.texture_image = Image.open(self.texture_path)
            print("Texture loaded successfully!")
        except Exception as e:
            print(f"Error loading texture: {e}")
            self.texture_image = None
    
    def generate_speech_audio(self, text, output_path):
        """Generate speech audio from text"""
        try:
            engine = pyttsx3.init()
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            # Wait for file to be written
            for _ in range(10):
                if os.path.exists(output_path):
                    try:
                        with wave.open(output_path, 'rb') as w:
                            return True
                    except wave.Error:
                        time.sleep(0.2)
                else:
                    time.sleep(0.2)
            return False
        except Exception as e:
            print(f"Error generating speech: {e}")
            return False
    
    def apply_lip_sync_animation(self, frame_time, audio_duration):
        """Apply lip-sync animation based on frame time"""
        # Simple lip-sync: open mouth more during speech
        lip_intensity = 0.3 + 0.4 * np.sin(frame_time * 10)  # Oscillating mouth movement
        self.lip_sync_intensity = lip_intensity
    
    def apply_facial_expressions(self, frame_time):
        """Apply subtle facial expressions"""
        # Add slight head movement and expressions
        self.head_rotation = 0.1 * np.sin(frame_time * 0.5)
        self.expression_intensity = 0.1 * np.sin(frame_time * 0.3)
    
    def render_frame_with_texture(self, frame_path, frame_time):
        """Render a single frame with texture mapping"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            fig = plt.figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            # Apply animations to vertices
            animated_vertices = self.original_vertices.copy()
            
            # Apply lip-sync deformation (simplified)
            if self.lip_sync_intensity > 0:
                scale_factor = 1.0 + self.lip_sync_intensity * 0.1
                animated_vertices *= scale_factor
            
            # Apply head rotation
            if self.head_rotation != 0:
                cos_rot = np.cos(self.head_rotation)
                sin_rot = np.sin(self.head_rotation)
                for i in range(len(animated_vertices)):
                    x, y, z = animated_vertices[i]
                    animated_vertices[i] = [x * cos_rot - z * sin_rot, y, x * sin_rot + z * cos_rot]

            # Prepare faces for Poly3DCollection
            num_faces = min(20000, len(self.mesh.faces))  # Increase for visibility
            faces = []
            for i in range(num_faces):
                face = self.mesh.faces[i]
                face_vertices = [animated_vertices[vid] for vid in face]
                faces.append(face_vertices)

            # Create mesh collection with texture-like appearance
            if self.texture_image:
                # Use skin-tone colors to simulate texture
                mesh_collection = Poly3DCollection(faces, facecolor='#f5d0c5', edgecolor='#d4a574', alpha=0.9)
            else:
                mesh_collection = Poly3DCollection(faces, facecolor='#ccccff', edgecolor='gray', alpha=1.0)
            
            ax.add_collection3d(mesh_collection)

            # Auto-scale axes to fit the mesh
            all_xyz = np.array(animated_vertices)
            max_range = (all_xyz.max(axis=0) - all_xyz.min(axis=0)).max() / 2.0
            mid_x = (all_xyz[:,0].max() + all_xyz[:,0].min()) * 0.5
            mid_y = (all_xyz[:,1].max() + all_xyz[:,1].min()) * 0.5
            mid_z = (all_xyz[:,2].max() + all_xyz[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_box_aspect([1,1,1])
            
            # Set camera angle to look at the face
            ax.view_init(elev=-90, azim=0 + self.head_rotation * 30)
            
            # Remove grid and axes
            ax.set_axis_off()
            
            # Add title and info
            ax.set_title(f"3D Talking Head - Frame {self.frame_count}", color='white')
            ax.text2D(0.02, 0.98, f"Lip Sync: {self.lip_sync_intensity:.2f}", 
                     transform=ax.transAxes, color='white', fontsize=10)
            
            # Save the image with even dimensions
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='black', 
                       pad_inches=0, format='png')
            plt.close(fig)
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            print(f"Error rendering frame: {e}")
            return False
    
    def generate_talking_head_video(self, text, fps=30):
        """Generate a talking head video from text"""
        print(f"Generating textured 3D talking head video for text: '{text}'")
        
        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), "results", f"textured_talking_head_{uuid.uuid4().hex}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate speech audio
        audio_path = os.path.join(self.output_dir, "speech.wav")
        if not self.generate_speech_audio(text, audio_path):
            return "Failed to generate speech audio."
        
        # Get audio duration
        try:
            with wave.open(audio_path, 'rb') as w:
                audio_duration = w.getnframes() / w.getframerate()
        except:
            audio_duration = len(text) * 0.1  # Estimate duration
        
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Generate frames
        frame_paths = []
        total_frames = int(audio_duration * fps)
        
        print(f"Generating {total_frames} frames...")
        
        for frame_idx in range(total_frames):
            frame_time = frame_idx / fps
            
            # Apply animations
            self.apply_lip_sync_animation(frame_time, audio_duration)
            self.apply_facial_expressions(frame_time)
            
            # Render frame
            frame_path = os.path.join(self.output_dir, f"frame_{frame_idx:04d}.png")
            if self.render_frame_with_texture(frame_path, frame_time):
                frame_paths.append(frame_path)
            
            # Progress update
            if frame_idx % 10 == 0:
                print(f"Generated frame {frame_idx}/{total_frames}")
        
        # Assemble video
        video_path = os.path.join(self.output_dir, "talking_head_textured.mp4")
        if self.assemble_video(frame_paths, video_path, audio_path, fps):
            print(f"Video generated successfully: {video_path}")
            return video_path
        else:
            return "Failed to assemble video."
    
    def assemble_video(self, frame_paths, video_path, audio_path, fps):
        """Assemble frames into video with audio"""
        try:
            # Ensure frames have even dimensions
            self.ensure_even_dimensions(frame_paths)
            
            # Use ffmpeg to create video with audio
            cmd = [
                "ffmpeg", "-y",  # Overwrite output
                "-framerate", str(fps),
                "-i", os.path.join(self.output_dir, "frame_%04d.png"),
                "-i", audio_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
                "-shortest",  # End when shortest input ends
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error assembling video: {e}")
            return False
    
    def ensure_even_dimensions(self, frame_paths):
        """Ensure all frames have even dimensions for video encoding"""
        for frame_path in frame_paths:
            if os.path.exists(frame_path):
                try:
                    with Image.open(frame_path) as img:
                        width, height = img.size
                        # If dimensions are odd, resize to even
                        if width % 2 != 0 or height % 2 != 0:
                            new_width = width + (width % 2)
                            new_height = height + (height % 2)
                            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            resized_img.save(frame_path)
                            print(f"Resized frame to {new_width}x{new_height}")
                except Exception as e:
                    print(f"Error processing frame {frame_path}: {e}")

def main():
    # Model and texture paths
    obj_path = "Portrait_Study_0702105334_texture.obj"
    texture_path = "Portrait_Study_0702105334_texture.png"
    
    # Create talking head system
    talking_head = TexturedTalkingHead3D(obj_path, texture_path)
    
    # Test text
    test_text = "Hello! This is a test of the textured 3D model talking head system."
    
    print("Starting textured 3D model talking head generation...")
    video_path = talking_head.generate_talking_head_video(test_text, fps=15)
    
    if video_path and os.path.exists(video_path):
        print(f"✅ Success! Video saved to: {video_path}")
    else:
        print(f"❌ Error: {video_path}")

if __name__ == "__main__":
    main() 