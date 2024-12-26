from dataclasses import dataclass, field

import torch

# Registry to store case classes
class_registry = {}


def register_case(name: str):
    def decorator(cls):
        class_registry[name] = cls
        return cls

    return decorator

@dataclass
class BaseCase:
    name: str
    validators: list[str]
    base_validator: str
    num_epochs: int = 40
    reset_bonds: bool = False
    reset_bonds_index: int = None
    reset_bonds_epoch: int = None
    servers: list[str] = field(default_factory=lambda: ["Server 1", "Server 2"])

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        raise NotImplementedError(
            "Subclasses must implement the weights_epochs property."
        )

    @property
    def stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.8, 0.1, 0.1])] * self.num_epochs

    def __post_init__(self):
        if self.base_validator not in self.validators:
            raise ValueError(
                f"base_validator '{self.base_validator}' must be in validators list."
            )


def create_case(case_name: str, **kwargs) -> BaseCase:
    if case_name not in class_registry:
        raise ValueError(f"Case '{case_name}' is not registered.")
    case_class = class_registry[case_name]
    return case_class(**kwargs)


@register_case("Case 1")
@dataclass
class Case1(BaseCase):
    name: str = "Case 1 - kappa moves first"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small lazy vali. (0.1)",
            "Small lazier vali. (0.1)",
        ]
    )
    base_validator: str = "Big vali. (0.8)"

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_1 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 1:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 0] = 1.0  # Validator B -> Server 1
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif epoch == 2:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            else:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            weights_epochs_case_1.append(W)
        return weights_epochs_case_1


@register_case("Case 2")
@dataclass
class Case2(BaseCase):
    name: str = "Case 2 - kappa moves second"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_2 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 1:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif epoch == 2:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            else:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            weights_epochs_case_2.append(W)
        return weights_epochs_case_2


@register_case("Case 3")
@dataclass
class Case3(BaseCase):
    name: str = "Case 3 - kappa moves third"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_3 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 1:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif epoch == 2:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            else:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            weights_epochs_case_3.append(W)
        return weights_epochs_case_3


@register_case("Case 4")
@dataclass
class Case4(BaseCase):
    name: str = "Case 4 - all validators switch"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small vali. (0.1)",
            "Small vali 2. (0.1)",
        ]
    )
    base_validator: str = "Big vali. (0.8)"

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_4 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # All validators support Server 1
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 0] = 1.0  # Validator B -> Server 1
                W[2, 0] = 1.0  # Validator C -> Server 1
            if epoch >= 1:
                # All validators support Server 2
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            weights_epochs_case_4.append(W)
        return weights_epochs_case_4


@register_case("Case 5")
@dataclass
class Case5(BaseCase):
    name: str = "Case 5 - kappa moves second, then third"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager-eager vali. (0.1)",
            "Small eager-lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager-eager vali. (0.1)"
    reset_bonds: bool = True
    reset_bonds_index: int = 1
    reset_bonds_epoch: int = 20

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_5 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 1:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif epoch == 2:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif 3 <= epoch <= 20:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            elif epoch == 21:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 0] = 1.0  # Validator B -> Server 1
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif epoch == 22:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 0] = 1.0  # Validator B -> Server 1
                W[2, 0] = 1.0  # Validator C -> Server 1
            else:
                # Subsequent epochs
                W[:, 0] = 1.0  # All validators -> Server 1
            weights_epochs_case_5.append(W)
        return weights_epochs_case_5


@register_case("Case 6")
@dataclass
class Case6(BaseCase):
    name: str = "Case 6 - kappa moves second, then all validators switch"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"
    reset_bonds: bool = True
    reset_bonds_index: int = 0
    reset_bonds_epoch: int = 21

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_6 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # All validators support Server 1
                W[:, 0] = 1.0
            elif epoch == 1:
                # Validator B switches to Server 2
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif epoch == 2:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif 3 <= epoch <= 20:
                # All validators support Server 2
                W[:, 1] = 1.0
            else:
                # All validators switch back to Server 1
                W[:, 0] = 1.0
            weights_epochs_case_6.append(W)
        return weights_epochs_case_6


@register_case("Case 7")
@dataclass
class Case7(BaseCase):
    name: str = "Case 7 - big vali moves late, then all but one small vali moves late"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small eager-lazy vali. (0.1)",
            "Small eager-eager vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager-eager vali. (0.1)"
    reset_bonds: bool = True
    reset_bonds_index: int = 0
    reset_bonds_epoch: int = 21

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_7 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 1:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif epoch == 2:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif 3 <= epoch <= 20:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            elif epoch == 21:
                W[0, 1] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 1
                W[2, 0] = 1.0  # Validator C -> Server 2
            else:
                # Subsequent epochs
                W[:, 0] = 1.0  # All validators -> Server 1
            weights_epochs_case_7.append(W)
        return weights_epochs_case_7

@register_case("Case 8")
@dataclass
class Case8(BaseCase):
    name: str = "Case 8 - big vali moves late, then late"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big dishonest lazy vali. (0.8)",
            "Small eager-eager vali. (0.1)",
            "Small eager-eager vali 2. (0.1)",
        ]
    )
    base_validator: str = "Small eager-eager vali. (0.1)"
    reset_bonds: bool = True
    reset_bonds_index: int = 1
    reset_bonds_epoch: int = 20

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_8 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                W[:, 0] = 1.0
            elif epoch == 1:
                # Validators B and C switch to Server 2
                W[0, 0] = 1.0  # Validator A
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif 2 <= epoch <= 20:
                # Validator A copies weights but still supports Server 1 with minimal weight
                W[0, 1] = 1.0  # Validator A
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 1] = 1.0  # Validator C -> Server 2
            elif epoch == 21:
                # Validators B and C switch back to Server 1
                W[0, 1] = 1.0  # Validator A
                W[1, 0] = 1.0  # Validator B -> Server 1
                W[2, 0] = 1.0  # Validator C -> Server 1
            else:
                W[:, 0] = 1.0
            weights_epochs_case_8.append(W)
        return weights_epochs_case_8

@register_case("Case 9")
@dataclass
class Case9(BaseCase):
    name: str = "Case 9 - small validators merged in e5"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small vali. (0.1/0.2)",
            "Small vali 2. (0.1/0.0)",
        ]
    )
    base_validator: str = "Big vali. (0.8)"

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_9 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            W[:, 1] = 1.0  # All validators -> Server 2
            weights_epochs_case_9.append(W)
        return weights_epochs_case_9

    @property
    def stakes_epochs(self) -> list[torch.Tensor]:
        stakes_epochs_case_9 = []
        for epoch in range(self.num_epochs):
            if 0 <= epoch <= 5:
                stakes = torch.tensor([0.8, 0.1, 0.1])
            else:
                stakes = torch.tensor([0.8, 0.2, 0.0])  # Validator C joins Validator B
            stakes_epochs_case_9.append(stakes)
        return stakes_epochs_case_9


@register_case("Case 10")
@dataclass
class Case10(BaseCase):
    name: str = "Case 10 - kappa delayed"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big delayed vali. (0.8)",
            "Small eager vali. (0.1)",
            "Small lazy vali. (0.1)",
        ]
    )
    base_validator: str = "Small eager vali. (0.1)"

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_10 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # Initially, consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif 1 <= epoch < 10:
                W[0, 0] = 1.0  # Validator A -> Server 1
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            elif epoch == 10:
                W[0, 1] = 1.0  # Validator A -> Server 2
                W[1, 1] = 1.0  # Validator B -> Server 2
                W[2, 0] = 1.0  # Validator C -> Server 1
            else:
                # Subsequent epochs
                W[:, 1] = 1.0  # All validators -> Server 2
            weights_epochs_case_10.append(W)
        return weights_epochs_case_10


@register_case("Case 11")
@dataclass
class Case11(BaseCase):
    name: str = "Case 11 - clipping demo"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. 1 (0.49)",
            "Big vali. 2 (0.49)",
            "Small vali. (0.02)",
        ]
    )
    base_validator: str = "Big vali. 1 (0.49)"
    reset_bonds: bool = True
    reset_bonds_index: int = 1
    reset_bonds_epoch: int = 20

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_11 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch < 20:
                # Server 1
                W[0, 0] = 0.3
                W[1, 0] = 0.6
                W[2, 0] = 0.61
                # Server 2
                W[0, 1] = 0.7
                W[1, 1] = 0.4
                W[2, 1] = 0.39
            else:
                # Server 1
                W[0, 0] = 0.3
                W[1, 0] = 0.6
                W[2, 0] = 0.3
                # Server 2
                W[0, 1] = 0.7
                W[1, 1] = 0.4
                W[2, 1] = 0.61
            weights_epochs_case_11.append(W)
        return weights_epochs_case_11

    @property
    def stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.49, 0.49, 0.02])] * self.num_epochs


@register_case("Case 12")
@dataclass
class Case12(BaseCase):
    name: str = "Case 12 - all validators switch, but small validator/s support alt miner with minimal weight"
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small dishonest vali. (0.1)",
            "Small vali. (0.1)",
        ]
    )
    base_validator: str = "Big vali. (0.8)"
    reset_bonds: bool = True
    reset_bonds_index: int = 1
    reset_bonds_epoch: int = 20

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_12 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch == 0:
                # All Validators support server 1
                W[0, 0] = 1.0
                W[1, :] = torch.tensor(
                    [0.999, 0.001]
                )  # Small dishonest vali. shifts slightly to Server 2
                W[2, 0] = 1.0
            elif 1 <= epoch <= 20:
                # All Validators support server 2
                W[0, 1] = 1.0
                W[1, :] = torch.tensor(
                    [0.001, 0.999]
                )  # Small dishonest vali. shifts back to Server 2
                W[2, 1] = 1.0
            else:
                # All Validators support server 1
                W[0, 0] = 1.0
                W[1, :] = torch.tensor([0.999, 0.001])
                W[2, 0] = 1.0
            weights_epochs_case_12.append(W)
        return weights_epochs_case_12


@dataclass
@register_case("Case 13")
class Case13(BaseCase):
    name: str = (
        "Case 13 - Big vali supports server 2, small validator/s support server 1"
    )
    validators: list[str] = field(
        default_factory=lambda: [
            "Big vali. (0.8)",
            "Small vali. (0.1)",
            "Small vali 2. (0.1)",
        ]
    )
    base_validator: str = "Big vali. (0.8)"
    reset_bonds: bool = True
    reset_bonds_index: int = 0
    reset_bonds_epoch: int = 20

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_13 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch <= 20:
                W[0, 1] = 1.0  # Big vali. supports Server 2
                W[1, :] = torch.tensor([0.5, 0.5])  # Small vali. supports Server 1
                W[2, 1] = 1.0  # Small vali 2. supports Server 2
            else:
                W[0, 1] = 1.0  # Big vali. continues to support Server 2
                W[1, :] = torch.tensor([0.5, 0.5])  # Small vali. supports Server 1
                W[2, :] = torch.tensor([0.5, 0.5])  # Small vali 2. supports Server 1
            weights_epochs_case_13.append(W)
        return weights_epochs_case_13


@dataclass
@register_case("Case 14")
class Case14(BaseCase):
    name: str = "Case 14 - All validators support Server 1, one of them switches to Server 2 for one epoch"
    validators: list[str] = field(
        default_factory=lambda: ["Vali. 1 (0.33)", "Vali. 2 (0.33)", "Vali. 3 (0.34)"]
    )
    base_validator: str = "Vali. 1 (0.33)"
    reset_bonds: bool = False

    @property
    def weights_epochs(self) -> list[torch.Tensor]:
        weights_epochs_case_14 = []
        for epoch in range(self.num_epochs):
            W = torch.zeros(3, 2)
            if epoch >= 0 and epoch < 20:
                # Consensus is achieved by all Validators
                W[:, 0] = 1.0
            elif epoch == 20:
                W[0, 0] = 1.0  # Validator 1 -> Server 1
                W[1, 0] = 1.0  # Validator 2 -> Server 1
                W[2, 1] = 1.0  # Validator 3 -> Server 2
            else:
                W[:, 0] = 1.0  # All validators -> Server 1
            weights_epochs_case_14.append(W)
        return weights_epochs_case_14

    @property
    def stakes_epochs(self) -> list[torch.Tensor]:
        return [torch.tensor([0.33, 0.33, 0.34])] * self.num_epochs


# Instantiate all cases dynamically
cases = [cls() for cls in class_registry.values()]

# Example Usage
if __name__ == "__main__":
    for case in cases:
        print(f"--- {case.name} ---")
        print("Validators:", case.validators)
        print("Base Validator:", case.base_validator)
        print("Reset Bonds:", case.reset_bonds)
        if case.reset_bonds:
            print("Reset Bonds Index:", case.reset_bonds_index)
            print("Reset Bonds Epoch:", case.reset_bonds_epoch)
        print("Weights for first 3 epochs:")
        for i in range(3):
            print(f"Epoch {i}:")
            print(case.weights_epochs[i])
        print("Stakes for first 3 epochs:")
        for i in range(3):
            print(f"Epoch {i}: {case.stakes_epochs[i]}")
        print("\n")
