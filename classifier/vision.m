// Reads oriented PNG pixels from stdin. Coordinates use a top-left origin.
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
int main(void) {
    @autoreleasepool {
        NSData *image = [[NSFileHandle fileHandleWithStandardInput] readDataToEndOfFile];
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.revision = VNRecognizeTextRequestRevision3;
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.recognitionLanguages = @[@"en-US"];
        request.usesLanguageCorrection = YES;
        NSError *error = nil;
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithData:image options:@{}];
        if (![handler performRequests:@[request] error:&error]) {
            fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        NSMutableArray *lines = [NSMutableArray array];
        for (VNRecognizedTextObservation *line in request.results) {
            VNRecognizedText *text = [line topCandidates:1].firstObject;
            if (!text) continue;
            CGRect b = line.boundingBox;
            [lines addObject:@{@"text":text.string, @"confidence":@(text.confidence),
                @"box":@[@(b.origin.x), @(1-CGRectGetMaxY(b)), @(b.size.width), @(b.size.height)]}];
        }
        NSData *json = [NSJSONSerialization dataWithJSONObject:lines options:0 error:&error];
        [[NSFileHandle fileHandleWithStandardOutput] writeData:json];
    }
    return 0;
}
