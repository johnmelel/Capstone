from manager import AgentManager
from PIL import Image

if __name__ == '__main__':
    manager = AgentManager()
    query = 'Find diabetic foot ulcer cases with MRI images'
    example_img = Image.new('RGB', (224,224))
    manager.run(query, [example_img])
    print('Done')
