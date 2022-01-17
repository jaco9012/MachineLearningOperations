import torch
import torchvision.models as models


randImg = torch.rand(64, 3, 7, 7)

resnet18 = models.resnet18(pretrained=True)
script_model = torch.jit.script(resnet18)
values_resnet18, indecies_resnet18 = torch.topk(resnet18(randImg), 5)
values_script_model, indecies_script_model = torch.topk(script_model(randImg), 5)
assert torch.allclose(values_resnet18, values_script_model), "does not give the same output"
script_model.save('PersonalChanges/MachineLearningOperations/cookiecutter/model_store/deployable_model.pt')

