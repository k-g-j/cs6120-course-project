import logging
import os
import sys
from pathlib import Path


def setup_logging():
    """Configure logging for directory setup."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class ProjectSetup:
    """Handle project directory setup with proper error handling."""

    def __init__(self):
        self.logger = setup_logging()
        self.base_dir = Path.cwd()

        # Define directory structure
        self.directory_structure = {
            'data': ['solar_data'],
            'logs': [],
            'results': ['ablation_studies', 'ensemble'],
            'models': ['ensemble', 'baseline', 'checkpoints'],
            'processed_data': [],
            'reports': [],
            'visualizations': ['ensemble', 'ablation', 'final_analysis'],
            'src': [
                'final_analysis',
                'models',
                'visualization'
            ]
        }

    def create_directory(self, path):
        """Create a directory with error handling."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✓ Created/verified directory: {path}")
            return True
        except PermissionError:
            self.logger.error(f"✗ Permission denied creating directory: {path}")
            return False
        except Exception as e:
            self.logger.error(f"✗ Error creating directory {path}: {str(e)}")
            return False

    def verify_directory_permissions(self, path):
        """Verify directory permissions."""
        try:
            # Try to create a temporary file
            test_file = path / '.permissions_test'
            test_file.touch()
            test_file.unlink()  # Remove test file
            return True
        except Exception as e:
            self.logger.error(f"✗ Permission verification failed for {path}: {str(e)}")
            return False

    def setup_project_structure(self):
        """Set up the complete project directory structure."""
        self.logger.info("Starting project directory setup...")

        success = True
        created_dirs = []

        try:
            for main_dir, subdirs in self.directory_structure.items():
                main_path = self.base_dir / main_dir

                # Create main directory
                if self.create_directory(main_path):
                    created_dirs.append(main_path)

                    # Verify permissions
                    if not self.verify_directory_permissions(main_path):
                        success = False
                        continue

                    # Create subdirectories
                    for subdir in subdirs:
                        subdir_path = main_path / subdir
                        if self.create_directory(subdir_path):
                            created_dirs.append(subdir_path)

                            # Verify permissions
                            if not self.verify_directory_permissions(subdir_path):
                                success = False
                else:
                    success = False

            if success:
                self.logger.info("✓ Project directory structure created successfully")
            else:
                self.logger.warning("⚠ Project setup completed with some errors")

            # Create empty __init__.py files in Python directories
            self._create_init_files()

            return success, created_dirs

        except Exception as e:
            self.logger.error(f"✗ Critical error during setup: {str(e)}")
            return False, created_dirs

    def _create_init_files(self):
        """Create __init__.py files in Python package directories."""
        try:
            python_dirs = [
                self.base_dir / 'src',
                self.base_dir / 'src' / 'final_analysis',
                self.base_dir / 'src' / 'models',
                self.base_dir / 'src' / 'visualization'
            ]

            for dir_path in python_dirs:
                if dir_path.exists():
                    init_file = dir_path / '__init__.py'
                    if not init_file.exists():
                        init_file.touch()
                        self.logger.info(f"✓ Created {init_file}")

        except Exception as e:
            self.logger.error(f"✗ Error creating __init__.py files: {str(e)}")

    def verify_setup(self, created_dirs):
        """Verify the setup was successful."""
        self.logger.info("\nVerifying project setup...")

        all_valid = True
        for directory in created_dirs:
            if not directory.exists():
                self.logger.error(f"✗ Directory missing: {directory}")
                all_valid = False
            elif not os.access(directory, os.W_OK):
                self.logger.error(f"✗ Directory not writable: {directory}")
                all_valid = False
            else:
                self.logger.info(f"✓ Verified {directory}")

        return all_valid


def main():
    """Run the project setup."""
    setup = ProjectSetup()
    success, created_dirs = setup.setup_project_structure()

    if success:
        if setup.verify_setup(created_dirs):
            setup.logger.info("\n✓ Project setup completed successfully")
        else:
            setup.logger.error("\n✗ Project setup verification failed")
    else:
        setup.logger.error("\n✗ Project setup failed")


if __name__ == "__main__":
    main()
